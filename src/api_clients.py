"""
API Integration Examples

This module demonstrates how to properly use API keys loaded from .env
for various external services.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class APIClient:
    """Base API client with retry logic and error handling."""
    
    def __init__(self, base_url: str, api_key: str, service_name: str):
        self.base_url = base_url
        self.api_key = api_key
        self.service_name = service_name
        self.client = httpx.AsyncClient(
            timeout=settings.api_timeout,
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a request with retry logic."""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"{self.service_name} API error: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class OpenAIClient(APIClient):
    """OpenAI API client using API key from .env."""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key,
            service_name="OpenAI"
        )
    
    async def generate_embedding(self, text: str) -> list:
        """Generate embedding for text."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Check OPENAI_API_KEY in .env")
        
        response = await self.make_request(
            "POST",
            "embeddings",
            json={
                "input": text,
                "model": "text-embedding-ada-002"
            }
        )
        return response["data"][0]["embedding"]
    
    async def analyze_text(self, text: str, prompt: str) -> str:
        """Analyze text using GPT."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Check OPENAI_API_KEY in .env")
        
        response = await self.make_request(
            "POST",
            "chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 1000
            }
        )
        return response["choices"][0]["message"]["content"]


class AnthropicClient(APIClient):
    """Anthropic Claude API client using API key from .env."""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.anthropic.com/v1",
            api_key=settings.anthropic_api_key,
            service_name="Anthropic"
        )
        # Override headers for Anthropic
        self.client.headers.update({
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        })
        # Remove Authorization header as Anthropic uses x-api-key
        self.client.headers.pop("Authorization", None)
    
    async def analyze_text(self, text: str, prompt: str) -> str:
        """Analyze text using Claude."""
        if not self.api_key:
            raise ValueError("Anthropic API key not configured. Check ANTHROPIC_API_KEY in .env")
        
        response = await self.make_request(
            "POST",
            "messages",
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n{text}"
                    }
                ]
            }
        )
        return response["content"][0]["text"]


class GeminiClient:
    """Google Gemini API client using API key from .env."""
    
    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.client = httpx.AsyncClient(timeout=settings.api_timeout)
    
    async def analyze_text(self, text: str, prompt: str) -> str:
        """Analyze text using Gemini."""
        if not self.api_key:
            raise ValueError("Gemini API key not configured. Check GEMINI_API_KEY in .env")
        
        url = f"{self.base_url}/models/gemini-1.5-flash:generateContent"
        
        try:
            response = await self.client.post(
                url,
                params={"key": self.api_key},
                json={
                    "contents": [{
                        "parts": [{
                            "text": f"{prompt}\n\n{text}"
                        }]
                    }]
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPError as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class FirecrawlClient(APIClient):
    """Firecrawl API client using API key from .env."""
    
    def __init__(self):
        super().__init__(
            base_url="https://api.firecrawl.dev/v0",
            api_key=settings.firecrawl_api_key,
            service_name="Firecrawl"
        )
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a URL using Firecrawl."""
        if not self.api_key:
            raise ValueError("Firecrawl API key not configured. Check FIRECRAWL_API_KEY in .env")
        
        response = await self.make_request(
            "POST",
            "scrape",
            json={
                "url": url,
                "pageOptions": {
                    "onlyMainContent": True
                }
            }
        )
        return response["data"]


class MultiModelAnalyzer:
    """Multi-model analyzer that uses all available APIs."""
    
    def __init__(self):
        self.openai = OpenAIClient()
        self.anthropic = AnthropicClient()
        self.gemini = GeminiClient()
        self.firecrawl = FirecrawlClient()
    
    async def analyze_with_multiple_models(self, text: str, prompt: str) -> Dict[str, str]:
        """Analyze text with multiple models and return results."""
        results = {}
        
        # Try each model, but don't fail if one is missing API key
        models = [
            ("openai", self.openai.analyze_text),
            ("anthropic", self.anthropic.analyze_text),
            ("gemini", self.gemini.analyze_text),
        ]
        
        for model_name, analyze_func in models:
            try:
                result = await analyze_func(text, prompt)
                results[model_name] = result
                logger.info(f"Successfully analyzed with {model_name}")
            except ValueError as e:
                logger.warning(f"Skipping {model_name}: {e}")
                results[model_name] = f"Not available: {e}"
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                results[model_name] = f"Error: {e}"
        
        return results
    
    async def research_topic(self, topic: str, urls: Optional[list] = None) -> Dict[str, Any]:
        """Research a topic using web scraping and analysis."""
        research_results = {
            "topic": topic,
            "scraped_content": [],
            "analysis": {}
        }
        
        # Scrape URLs if provided
        if urls:
            for url in urls:
                try:
                    content = await self.firecrawl.scrape_url(url)
                    research_results["scraped_content"].append({
                        "url": url,
                        "content": content
                    })
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {e}")
                    research_results["scraped_content"].append({
                        "url": url,
                        "error": str(e)
                    })
        
        # Analyze scraped content
        if research_results["scraped_content"]:
            combined_content = "\n\n".join([
                item.get("content", {}).get("markdown", "")
                for item in research_results["scraped_content"]
                if "content" in item
            ])
            
            if combined_content:
                prompt = f"Analyze the following content related to '{topic}' and provide key insights:"
                research_results["analysis"] = await self.analyze_with_multiple_models(
                    combined_content[:5000],  # Limit content length
                    prompt
                )
        
        return research_results
    
    async def close(self):
        """Close all API clients."""
        await asyncio.gather(
            self.openai.close(),
            self.anthropic.close(),
            self.gemini.close(),
            self.firecrawl.close(),
            return_exceptions=True
        )


# Example usage function
async def example_usage():
    """Example of how to use the API clients with .env configuration."""
    print("=== API Integration Example ===")
    
    # Create analyzer
    analyzer = MultiModelAnalyzer()
    
    try:
        # Example text analysis
        sample_text = "This is a sample legal document about contract law."
        prompt = "Analyze this text and identify key legal concepts."
        
        print("Analyzing sample text with multiple models...")
        results = await analyzer.analyze_with_multiple_models(sample_text, prompt)
        
        for model, result in results.items():
            print(f"\n{model.upper()} Result:")
            print(f"  {result[:100]}..." if len(result) > 100 else f"  {result}")
        
        # Example research
        print("\nTesting web scraping...")
        research = await analyzer.research_topic(
            "legal case analysis",
            ["https://example.com"]  # Replace with actual URLs
        )
        
        print(f"Research completed for: {research['topic']}")
        print(f"Scraped {len(research['scraped_content'])} URLs")
        
    except Exception as e:
        print(f"Example failed: {e}")
    
    finally:
        await analyzer.close()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
