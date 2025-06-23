# Persistent AI Analysis System - FastAPI + Async SQLAlchemy Template

A production-ready template for building continuously running, knowledge-accumulating AI analysis systems using modern 2024-2025 best practices.

## Architecture Overview

This template provides a robust foundation for:
- 📈 **Continuous Learning**: System that accumulates knowledge over time
- 🔄 **Persistent State**: All analysis results and insights stored in database
- ⚡ **High Performance**: Async FastAPI + SQLAlchemy for concurrent processing
- 🛡️ **Production Ready**: Docker, monitoring, error handling, and scalability
- 🔗 **API Integration**: Reliable external API integration with retry logic
- 📊 **Background Tasks**: Long-running analysis jobs with Celery/Redis

## Key Features

- **Async-First Architecture**: Full async/await pattern throughout
- **Persistent Knowledge Base**: All insights and analysis stored in PostgreSQL
- **Modular Design**: Clean separation of concerns with dependency injection
- **Robust Error Handling**: Circuit breakers, retries, and graceful degradation
- **Scalable Background Tasks**: Celery with Redis for distributed processing
- **Monitoring & Observability**: Prometheus metrics, structured logging
- **Container-Ready**: Docker composition for easy deployment

## Tech Stack

### Core Framework
- **FastAPI**: Async web framework with automatic OpenAPI docs
- **SQLAlchemy 2.0+**: Async ORM with connection pooling
- **PostgreSQL**: Primary database with asyncpg driver
- **Alembic**: Database migrations

### Background Processing
- **Celery**: Distributed task queue for long-running jobs
- **Redis**: Message broker and caching layer
- **APScheduler**: Periodic task scheduling

### External Integrations
- **httpx**: Async HTTP client with retry logic
- **Pydantic**: Data validation and settings management

### Production & Monitoring
- **Docker**: Containerization
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **Gunicorn + Uvicorn**: Production ASGI server

## Project Structure

```
src/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration management
├── database.py            # Database connection and session management
├── dependencies.py        # Dependency injection providers
├── middleware.py          # Custom middleware (CORS, logging, etc.)
├── exceptions.py          # Custom exception handlers
├── 
├── models/                # SQLAlchemy models
│   ├── __init__.py
│   ├── base.py           # Base model class
│   ├── document.py       # Document analysis models
│   ├── insight.py        # Knowledge and insights models
│   └── task.py           # Background task tracking
│
├── schemas/              # Pydantic schemas
│   ├── __init__.py
│   ├── document.py
│   ├── insight.py
│   └── task.py
│
├── api/                  # API routes
│   ├── __init__.py
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── documents.py  # Document upload/analysis endpoints
│   │   ├── insights.py   # Knowledge retrieval endpoints
│   │   ├── analysis.py   # Analysis control endpoints
│   │   └── health.py     # Health check endpoints
│
├── services/             # Business logic layer
│   ├── __init__.py
│   ├── document_service.py
│   ├── analysis_service.py
│   ├── insight_service.py
│   └── external_apis.py  # External API integrations
│
├── tasks/                # Celery background tasks
│   ├── __init__.py
│   ├── celery_app.py
│   ├── document_tasks.py
│   └── analysis_tasks.py
│
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── logging.py
│   ├── retry.py
│   └── metrics.py
│
└── tests/                # Test suite
    ├── __init__.py
    ├── conftest.py       # Pytest configuration
    ├── test_api/
    ├── test_services/
    └── test_tasks/
```

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone <this-template>
   cd fastapi-ai-system
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start Development Environment**
   ```bash
   docker-compose up -d
   # This starts PostgreSQL, Redis, and your app
   ```

3. **Run Migrations**
   ```bash
   docker-compose exec app alembic upgrade head
   ```

4. **Access the Application**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Grafana: http://localhost:3000

## Configuration

All configuration is managed through environment variables and Pydantic settings:

```python
# config.py
class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://user:pass@localhost/db"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # External APIs
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Performance
    db_pool_size: int = 20
    max_overflow: int = 0
    
    class Config:
        env_file = ".env"
```

## Best Practices Implemented

### 1. Database Session Management
- Async session factory with dependency injection
- Context-managed session lifecycle
- Proper connection pooling and cleanup

### 2. Background Task Processing
- Celery for distributed, fault-tolerant task execution
- Task result persistence in database
- Retry logic with exponential backoff

### 3. External API Integration
- Async HTTP client with circuit breaker pattern
- Request/response logging and metrics
- Rate limiting and quota management

### 4. Error Handling & Monitoring
- Structured logging with correlation IDs
- Prometheus metrics for all operations
- Health checks for dependencies

### 5. Testing Strategy
- Async test client for API testing
- Database fixtures for clean test state
- Mock external API dependencies

## Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
# Using Docker Swarm or Kubernetes
docker-compose -f docker-compose.prod.yml up -d
```

## Monitoring

The template includes a complete observability stack:

- **Metrics**: Prometheus scrapes metrics from `/metrics` endpoint
- **Dashboards**: Grafana with pre-configured dashboards
- **Logs**: Structured JSON logging with correlation IDs
- **Health Checks**: `/health` endpoint monitors all dependencies

## Scaling Considerations

1. **Horizontal Scaling**: Stateless app design allows multiple instances
2. **Database**: Connection pooling and read replicas
3. **Background Tasks**: Celery workers can be scaled independently
4. **Caching**: Redis for session storage and computed results
5. **Load Balancing**: Nginx reverse proxy for production

## Security Features

- CORS configuration
- Request rate limiting
- Input validation with Pydantic
- SQL injection prevention via ORM
- Secrets management via environment variables

---

This template provides a solid foundation for building production-ready, continuously learning AI analysis systems. It incorporates the latest 2024-2025 best practices while remaining flexible for your specific use case.
