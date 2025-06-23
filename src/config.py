"""
Configuration Management

Centralized configuration using Pydantic settings with environment variable support.
Follows 12-factor app principles for configuration management.
"""

import secrets
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application - All loaded from .env file
    app_name: str = Field("AI Analysis System", env="APP_NAME")
    version: str = Field("1.0.0", env="VERSION")
    debug: bool = Field(False, env="DEBUG")
    
    # Server - All loaded from .env file
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    workers: int = Field(4, env="WORKERS")
    
    # Security - All loaded from .env file
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    allowed_hosts: str = Field("*", env="ALLOWED_HOSTS")
    cors_origins: str = Field("*", env="CORS_ORIGINS")
    
    # Database - All loaded from .env file
    database_url: str = Field("postgresql+asyncpg://user:password@localhost:5432/ai_analysis", env="DATABASE_URL")
    database_pool_size: int = Field(20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(0, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT")
    database_pool_recycle: int = Field(3600, env="DATABASE_POOL_RECYCLE")
    
    # Redis - All loaded from .env file
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    
    # Celery - All loaded from .env file
    celery_broker_url: str = Field("redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field("redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    celery_task_serializer: str = Field("json", env="CELERY_TASK_SERIALIZER")
    celery_result_serializer: str = Field("json", env="CELERY_RESULT_SERIALIZER")
    celery_accept_content: str = Field("json", env="CELERY_ACCEPT_CONTENT")
    celery_timezone: str = Field("UTC", env="CELERY_TIMEZONE")
    
    # External APIs - All loaded from .env file
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")
    firecrawl_api_key: str = Field("", env="FIRECRAWL_API_KEY")
    
    # API Configuration - All loaded from .env file
    api_timeout: float = Field(30.0, env="API_TIMEOUT")
    api_max_retries: int = Field(3, env="API_MAX_RETRIES")
    api_retry_delay: float = Field(1.0, env="API_RETRY_DELAY")
    api_backoff_factor: float = Field(2.0, env="API_BACKOFF_FACTOR")
    
    # Rate Limiting - All loaded from .env file
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Background Tasks - All loaded from .env file
    enable_background_monitoring: bool = Field(True, env="ENABLE_BACKGROUND_MONITORING")
    analysis_batch_size: int = Field(10, env="ANALYSIS_BATCH_SIZE")
    analysis_timeout: int = Field(300, env="ANALYSIS_TIMEOUT")  # seconds
    
    # Logging - All loaded from .env file
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    
    # Monitoring - All loaded from .env file
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_namespace: str = Field("ai_analysis", env="METRICS_NAMESPACE")
    
    # File Storage - All loaded from .env file
    upload_dir: str = Field("/tmp/uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_file_types: str = Field(
        "application/pdf,text/plain,text/csv,application/json,text/html",
        env="ALLOWED_FILE_TYPES"
    )
    
    @validator("database_url")
    def validate_database_url(cls, v: str) -> str:
        """Ensure database URL uses async driver."""
        if not v.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
            raise ValueError("Database URL must use an async driver")
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v: Any) -> str:
        """Parse CORS origins from string."""
        if isinstance(v, list):
            return ",".join(v)
        return str(v)
    
    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v: Any) -> str:
        """Parse allowed hosts from string."""
        if isinstance(v, list):
            return ",".join(v)
        return str(v)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DatabaseSettings:
    """Database-specific settings derived from main settings."""
    
    def __init__(self, settings: Settings):
        self.url = settings.database_url
        self.pool_size = settings.database_pool_size
        self.max_overflow = settings.database_max_overflow
        self.pool_timeout = settings.database_pool_timeout
        self.pool_recycle = settings.database_pool_recycle
        
        # Connection arguments for asyncpg
        self.connect_args: Dict[str, Any] = {
            "server_settings": {
                "jit": "off",  # Disable JIT for better connection time
            },
        }


class CelerySettings:
    """Celery-specific settings derived from main settings."""
    
    def __init__(self, settings: Settings):
        self.broker_url = settings.celery_broker_url
        self.result_backend = settings.celery_result_backend
        self.task_serializer = settings.celery_task_serializer
        self.result_serializer = settings.celery_result_serializer
        self.accept_content = settings.celery_accept_content
        self.timezone = settings.celery_timezone
        
        # Additional Celery configuration
        self.task_track_started = True
        self.task_time_limit = 30 * 60  # 30 minutes
        self.task_soft_time_limit = 25 * 60  # 25 minutes
        self.worker_prefetch_multiplier = 1
        self.task_acks_late = True
        self.worker_disable_rate_limits = False
        
        # Result backend settings
        self.result_expires = 3600  # 1 hour
        self.result_persistent = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures we create the settings object only once
    and reuse it throughout the application lifecycle.
    """
    return Settings()


def get_database_settings() -> DatabaseSettings:
    """Get database-specific settings."""
    return DatabaseSettings(get_settings())


def get_celery_settings() -> CelerySettings:
    """Get Celery-specific settings."""
    return CelerySettings(get_settings())


def validate_api_keys() -> None:
    """
    Validate that required API keys are present.
    
    Raises:
        ValueError: If any required API keys are missing.
    """
    settings = get_settings()
    missing_keys = []
    
    # Only require keys that you actually have
    required_keys = [
        ('gemini_api_key', 'GEMINI_API_KEY'),
        ('firecrawl_api_key', 'FIRECRAWL_API_KEY'),
    ]
    
    # Optional keys - warn but don't fail
    optional_keys = [
        ('openai_api_key', 'OPENAI_API_KEY'),
        ('anthropic_api_key', 'ANTHROPIC_API_KEY'),
    ]
    
    for attr_name, env_name in required_keys:
        value = getattr(settings, attr_name, "")
        if not value or value.strip() == "":
            missing_keys.append(env_name)
    
    if missing_keys:
        raise ValueError(
            f"Missing required API keys in .env file: {', '.join(missing_keys)}\n"
            f"Please add these keys to your .env file."
        )
    
    # Log warnings for optional keys
    import logging
    logger = logging.getLogger(__name__)
    for attr_name, env_name in optional_keys:
        value = getattr(settings, attr_name, "")
        if not value or value.strip() == "":
            logger.warning(f"Optional API key {env_name} not set - some features may be limited")


def validate_configuration() -> None:
    """
    Validate the entire configuration.
    
    Raises:
        ValueError: If any configuration is invalid.
    """
    settings = get_settings()
    errors = []
    
    # Check API keys
    try:
        validate_api_keys()
    except ValueError as e:
        errors.append(str(e))
    
    # Check database URL
    if not settings.database_url:
        errors.append("DATABASE_URL is required")
    
    # Check Redis URL
    if not settings.redis_url:
        errors.append("REDIS_URL is required")
    
    # Check secret key in production
    if not settings.debug and (not settings.secret_key or len(settings.secret_key) < 32):
        errors.append("SECRET_KEY must be at least 32 characters in production")
    
    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + 
            "\n".join(f"- {error}" for error in errors)
        )


def print_configuration_status() -> None:
    """Print the current configuration status for debugging."""
    settings = get_settings()
    
    print("=== Configuration Status ===")
    print(f"App Name: {settings.app_name}")
    print(f"Version: {settings.version}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Host: {settings.host}")
    print(f"Port: {settings.port}")
    print()
    
    print("=== API Keys Status ===")
    api_keys = [
        ('OpenAI', settings.openai_api_key),
        ('Anthropic', settings.anthropic_api_key),
        ('Gemini', settings.gemini_api_key),
        ('Firecrawl', settings.firecrawl_api_key),
    ]
    
    for name, key in api_keys:
        status = "✓ Set" if key and key.strip() else "✗ Missing"
        masked_key = f"{key[:8]}..." if len(key) > 8 else "Not set"
        print(f"{name}: {status} ({masked_key})")
    
    print()
    print(f"Database URL: {'✓ Set' if settings.database_url else '✗ Missing'}")
    print(f"Redis URL: {'✓ Set' if settings.redis_url else '✗ Missing'}")
    print(f"Upload Directory: {settings.upload_dir}")
    print("=================================")
