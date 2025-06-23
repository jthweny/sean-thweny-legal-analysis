"""
Database Configuration and Session Management

Provides async database engine, session factory, and dependency injection
for FastAPI endpoints. Implements best practices for connection pooling
and session lifecycle management.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool

from src.config import get_database_settings
from src.models.base import Base

logger = logging.getLogger(__name__)

# Global variables for database components
engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


def create_engine() -> AsyncEngine:
    """
    Create async database engine with optimized settings.
    
    Returns:
        AsyncEngine: Configured async database engine
    """
    db_settings = get_database_settings()
    
    # Determine engine parameters based on database type
    if "sqlite" in db_settings.url:
        # SQLite configuration
        engine_kwargs = {
            "echo": False,
            "poolclass": NullPool,
            "connect_args": {"check_same_thread": False},
        }
    else:
        # PostgreSQL configuration
        engine_kwargs = {
            "echo": False,
            "poolclass": QueuePool,
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
        }
    
    return create_async_engine(db_settings.url, **engine_kwargs)


async def init_db() -> None:
    """
    Initialize the database connection and create tables.
    
    This function should be called once at application startup.
    """
    global engine, async_session_factory
    
    try:
        # Create engine
        engine = create_engine()
        
        # Create session factory
        async_session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_db() -> None:
    """Close database connections and clean up resources."""
    global engine, async_session_factory
    
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")
    
    engine = None
    async_session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get a database session for FastAPI endpoints.
    
    Yields:
        AsyncSession: Database session
        
    Raises:
        RuntimeError: If database is not initialized
    """
    if async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def configure_sqlite_pragmas(dbapi_connection, connection_record):
    """
    Configure SQLite-specific pragmas for optimal performance.
    
    Args:
        dbapi_connection: Raw database connection
        connection_record: SQLAlchemy connection record
    """
    with dbapi_connection.cursor() as cursor:
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys=ON")
        
        # Optimize SQLite performance
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")


def setup_sqlite_event_listeners(engine: AsyncEngine) -> None:
    """
    Setup SQLite-specific event listeners.
    
    Args:
        engine: Async database engine
    """
    if "sqlite" in str(engine.url):
        # Add pragma configuration for SQLite
        event.listen(
            engine.sync_engine,
            "connect",
            configure_sqlite_pragmas
        )
        
        logger.info("SQLite pragma configuration enabled")


# Context Manager Classes

class DatabaseSession:
    """
    Context manager for database sessions in async contexts.
    
    Example:
        async with DatabaseSession() as session:
            result = await session.execute(select(User))
    """
    
    def __init__(self):
        self.session: AsyncSession | None = None
    
    async def __aenter__(self) -> AsyncSession:
        """Enter the context and create a new session."""
        if async_session_factory is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")
        
        self.session = async_session_factory()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and close the session."""
        if self.session:
            if exc_type:
                # Rollback on exception
                await self.session.rollback()
            await self.session.close()


class DatabaseManager:
    """
    Database manager for initialization, health checks, and lifecycle management.
    """
    
    def __init__(self):
        self.session: AsyncSession | None = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        try:
            await init_db()
            self._initialized = True
            logger.info("DatabaseManager initialized successfully")
        except Exception as e:
            logger.error(f"DatabaseManager initialization failed: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections."""
        await close_db()
        self._initialized = False
        logger.info("DatabaseManager closed")
    
    def is_connected(self) -> bool:
        """Check if database is connected and initialized."""
        return self._initialized and engine is not None
    
    async def health_check(self) -> bool:
        """Perform a health check on the database connection."""
        try:
            if not self.is_connected():
                return False
            
            async with DatabaseSession() as session:
                await session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def __aenter__(self) -> AsyncSession:
        """Enter the context and create a new session."""
        if async_session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        self.session = async_session_factory()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and close the session."""
        if self.session:
            if exc_type:
                # Rollback on exception
                await self.session.rollback()
            await self.session.close()


# Utility functions for common database operations

async def execute_query(query: str, parameters: dict | None = None) -> list:
    """
    Execute a raw SQL query and return results.
    
    Args:
        query: SQL query string
        parameters: Query parameters
        
    Returns:
        List of row results
    """
    async with DatabaseSession() as session:
        result = await session.execute(query, parameters or {})
        return result.fetchall()


async def health_check() -> bool:
    """
    Check if database is healthy and accepting connections.
    
    Returns:
        bool: True if database is healthy, False otherwise
    """
    try:
        async with DatabaseSession() as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# Export commonly used components
__all__ = [
    "init_db",
    "close_db", 
    "get_session",
    "DatabaseSession",
    "DatabaseManager",
    "execute_query",
    "health_check",
]