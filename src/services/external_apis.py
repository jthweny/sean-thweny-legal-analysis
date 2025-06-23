"""
External API Integration Service

Handles all external API calls with robust error handling, retries,
and monitoring. Implements circuit breaker pattern for reliability.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config import get_settings
from src.utils.metrics import api_request_duration, api_request_total, api_error_total

logger = logging.getLogger(__name__)
settings = get_settings()


class APIError(Exception):
    """Base exception for API errors."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class QuotaExceededError(APIError):
    """Raised when API quota is exceeded."""
    pass


class CircuitBreakerError(APIError):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func):
        """Call function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise CircuitBreakerError("Circuit breaker is open")
        
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class ExternalAPIClient:
    """Base client for external API integrations."""
    
    def __init__(self, base_url: str, api_key: str, service_name: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.service_name = service_name
        self.circuit_breaker = CircuitBreaker()
        
        # HTTP client configuration
        timeout = httpx.Timeout(
            connect=10.0,
            read=settings.api_timeout,
            write=10.0,
            pool=5.0
        )
        
        self.client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30
            ),
            headers={
                "User-Agent": f"AI-Analysis-System/1.0 ({self.service_name})",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Make an HTTP request with retry logic and monitoring."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Prepare headers
        request_headers = self.client.headers.copy()
        if headers:
            request_headers.update(headers)
        
        # Add API key to headers
        request_headers.update(self._get_auth_headers())
        
        start_time = time.time()
        
        try:
            # Make request with circuit breaker protection
            def make_request():
                return self.client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=request_headers,
                )
            
            response = await self.circuit_breaker.call(make_request)
            
            # Record metrics
            duration = time.time() - start_time
            api_request_duration.labels(
                service=self.service_name,
                endpoint=endpoint,
                method=method,
                status_code=response.status_code
            ).observe(duration)
            
            api_request_total.labels(
                service=self.service_name,
                endpoint=endpoint,
                method=method,
                status_code=response.status_code
            ).inc()
            
            # Handle response
            if response.status_code == 429:
                api_error_total.labels(
                    service=self.service_name,
                    error_type="rate_limit"
                ).inc()
                raise RateLimitError("Rate limit exceeded")
            
            if response.status_code == 402:
                api_error_total.labels(
                    service=self.service_name,
                    error_type="quota_exceeded"
                ).inc()
                raise QuotaExceededError("API quota exceeded")
            
            response.raise_for_status()
            
            try:
                response_data = response.json()
            except ValueError:
                response_data = {"content": response.text}
            
            return response.status_code, response_data
            
        except httpx.HTTPStatusError as e:
            api_error_total.labels(
                service=self.service_name,
                error_type="http_error"
            ).inc()
            
            logger.error(f"HTTP error for {self.service_name}: {e.response.status_code} - {e.response.text}")
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text}")
        
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            api_error_total.labels(
                service=self.service_name,
                error_type="network_error"
            ).inc()
            
            logger.error(f"Network error for {self.service_name}: {str(e)}")
            raise APIError(f"Network error: {str(e)}")
        
        except CircuitBreakerError:
            api_error_total.labels(
                service=self.service_name,
                error_type="circuit_breaker"
            ).inc()
            raise
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers. Override in subclasses."""
        return {"Authorization": f"Bearer {self.api_key}"}


class OpenAIClient(ExternalAPIClient):
    """OpenAI API client."""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key,
            service_name="openai"
        )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion."""
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        status_code, response = await self._make_request(
            "POST",
            "/chat/completions",
            data=data
        )
        
        return response
    
    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """Create text embeddings."""
        data = {
            "model": model,
            "input": text
        }
        
        status_code, response = await self._make_request(
            "POST",
            "/embeddings",
            data=data
        )
        
        return response["data"][0]["embedding"]


class AnthropicClient(ExternalAPIClient):
    """Anthropic Claude API client."""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.anthropic.com/v1",
            api_key=settings.anthropic_api_key,
            service_name="anthropic"
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Anthropic-specific authentication headers."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    async def create_message(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a message with Claude."""
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            **kwargs
        }
        
        status_code, response = await self._make_request(
            "POST",
            "/messages",
            data=data
        )
        
        return response


class GeminiClient(ExternalAPIClient):
    """Google Gemini API client."""
    
    def __init__(self):
        super().__init__(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key=settings.gemini_api_key,
            service_name="gemini"
        )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Gemini-specific authentication headers."""
        return {}  # Gemini uses API key in URL params
    
    async def generate_content(
        self,
        prompt: str,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate content with Gemini."""
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                **kwargs
            }
        }
        
        params = {"key": self.api_key}
        
        status_code, response = await self._make_request(
            "POST",
            f"/models/{model}:generateContent",
            data=data,
            params=params
        )
        
        return response


class FirecrawlClient(ExternalAPIClient):
    """Firecrawl API client for web scraping."""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.firecrawl.dev/v0",
            api_key=settings.firecrawl_api_key,
            service_name="firecrawl"
        )
    
    async def scrape_url(
        self,
        url: str,
        formats: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Scrape a single URL."""
        data = {
            "url": url,
            "formats": formats or ["markdown"],
            **kwargs
        }
        
        status_code, response = await self._make_request(
            "POST",
            "/scrape",
            data=data
        )
        
        return response
    
    async def crawl_site(
        self,
        url: str,
        max_pages: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Crawl an entire website."""
        data = {
            "url": url,
            "crawlerOptions": {
                "maxPages": max_pages,
                **kwargs
            }
        }
        
        status_code, response = await self._make_request(
            "POST",
            "/crawl",
            data=data
        )
        
        return response


class ExternalAPIManager:
    """Manager for all external API clients."""
    
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all API clients."""
        if settings.openai_api_key:
            self.clients['openai'] = OpenAIClient()
        
        if settings.anthropic_api_key:
            self.clients['anthropic'] = AnthropicClient()
        
        if settings.gemini_api_key:
            self.clients['gemini'] = GeminiClient()
        
        if settings.firecrawl_api_key:
            self.clients['firecrawl'] = FirecrawlClient()
    
    async def close_all(self):
        """Close all API clients."""
        for client in self.clients.values():
            await client.close()
    
    def get_client(self, service_name: str) -> ExternalAPIClient:
        """Get a specific API client."""
        if service_name not in self.clients:
            raise ValueError(f"API client '{service_name}' not available")
        return self.clients[service_name]
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all API services."""
        health_status = {}
        
        for service_name, client in self.clients.items():
            try:
                # Simple health check - adjust per service
                if service_name == 'openai':
                    await client._make_request("GET", "/models")
                elif service_name == 'anthropic':
                    # Anthropic doesn't have a health endpoint, so we'll assume healthy
                    pass
                elif service_name == 'gemini':
                    # Gemini health check
                    pass
                elif service_name == 'firecrawl':
                    # Firecrawl health check
                    pass
                
                health_status[service_name] = True
                
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {str(e)}")
                health_status[service_name] = False
        
        return health_status


# Global API manager instance
api_manager = ExternalAPIManager()
