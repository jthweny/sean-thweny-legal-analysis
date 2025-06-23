"""
FastAPI Application Entry Point

This module sets up the main FastAPI application with all necessary middleware,
routers, and startup/shutdown events for a production-ready AI analysis system.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.v1 import analysis, documents, health, insights
from src.config import get_settings
from src.database import close_db, init_db
from src.exceptions import setup_exception_handlers
from src.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    RequestIDMiddleware,
)
from src.utils.logging import setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown events.
    
    Handles:
    - Database initialization
    - Cache warming
    - Background task startup
    - Graceful shutdown
    """
    # Startup
    setup_logging()
    await init_db()
    
    # Start background monitoring
    if settings.enable_background_monitoring:
        # Start periodic health checks, cache cleanup, etc.
        pass
    
    yield
    
    # Shutdown
    await close_db()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="AI Analysis System",
        description="Persistent AI document analysis and knowledge accumulation system",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Custom middleware (order matters!)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(
        health.router,
        prefix="/health",
        tags=["health"],
    )
    
    app.include_router(
        documents.router,
        prefix="/api/v1/documents",
        tags=["documents"],
    )
    
    app.include_router(
        insights.router,
        prefix="/api/v1/insights",
        tags=["insights"],
    )
    
    app.include_router(
        analysis.router,
        prefix="/api/v1/analysis",
        tags=["analysis"],
    )
    
    # Setup Prometheus metrics
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/health", "/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="fastapi_inprogress",
        inprogress_labels=True,
    )
    
    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics")
    
    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "message": "AI Analysis System",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_config=None,  # We handle logging ourselves
    )
