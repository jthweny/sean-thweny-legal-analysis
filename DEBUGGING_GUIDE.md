# Development & Debugging Guide - AI Analysis System

## 🚨 Common Issues & Solutions

This guide covers the most frequent bugs, errors, and debugging challenges you'll encounter when building continuously running AI analysis systems with FastAPI + async SQLAlchemy + Celery + Redis.

## 🔧 Development Setup & Best Practices

### 1. Local Development Environment

```bash
# Clone the template
git clone <your-repo>
cd fastapi-ai-system

# Setup environment
cp .env.example .env
# Edit .env with your API keys and configuration

# Start development stack
docker-compose up -d postgres redis
docker-compose up app celery_worker celery_flower

# Run migrations
docker-compose exec app alembic upgrade head

# Access services
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Flower: http://localhost:5555
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### 2. Development Tools Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks for code quality
pre-commit install

# Enable async debugging
export PYTHONASYNCIODEBUG=1

# Enable SQLAlchemy logging for connection debugging
export SQLALCHEMY_LOG_LEVEL=INFO
```

## 🐛 Common Issues & Debugging Strategies

### Issue #1: Database Connection Pool Exhaustion

**Symptoms:**
- `TimeoutError: QueuePool limit of size 20 reached`
- Slow database queries
- App hangs on database operations

**Root Causes:**
```python
# ❌ BAD: Session not properly closed
async def bad_function():
    session = async_session_factory()
    result = await session.execute(query)
    # Session never closed - LEAK!
    return result

# ❌ BAD: Exception not handled
async def bad_function_2():
    async with async_session_factory() as session:
        result = await session.execute(query)
        # If exception occurs here, rollback won't happen
        process_result(result)  # Might raise exception
        await session.commit()
```

**✅ Solutions:**
```python
# ✅ GOOD: Proper session management
async def good_function():
    async with async_session_factory() as session:
        try:
            result = await session.execute(query)
            await session.commit()
            return result
        except Exception:
            await session.rollback()
            raise

# ✅ GOOD: Using dependency injection in FastAPI
async def api_endpoint(db: AsyncSession = Depends(get_db)):
    # Session automatically managed by FastAPI
    result = await db.execute(query)
    return result
```

**Debugging Tools:**
```python
# Monitor connection pool status
from src.database import engine

async def check_pool_status():
    pool = engine.pool
    print(f"Pool size: {pool.size()}")
    print(f"Checked out: {pool.checkedout()}")
    print(f"Overflow: {pool.overflow()}")
    print(f"Invalid: {pool.invalidated()}")

# Enable connection logging
import logging
logging.getLogger('sqlalchemy.pool').setLevel(logging.DEBUG)
```

### Issue #2: Async/Await Deadlocks

**Symptoms:**
- FastAPI endpoints hang indefinitely
- Celery tasks never complete
- High CPU usage with no progress

**Root Causes:**
```python
# ❌ BAD: Blocking I/O in async function
async def bad_async():
    import time
    time.sleep(10)  # BLOCKS THE EVENT LOOP!
    return "done"

# ❌ BAD: Mixing sync and async database calls
async def bad_db_call():
    # Using sync session in async context
    session = Session()  # Sync session
    result = session.query(Model).all()  # Blocks event loop
    return result

# ❌ BAD: Not awaiting async calls
async def bad_await():
    result = await_some_async_function()  # Forgot await!
    return result
```

**✅ Solutions:**
```python
# ✅ GOOD: Use asyncio.sleep for delays
async def good_async():
    import asyncio
    await asyncio.sleep(10)  # Non-blocking
    return "done"

# ✅ GOOD: Proper async database usage
async def good_db_call():
    async with async_session_factory() as session:
        result = await session.execute(select(Model))
        return result.scalars().all()

# ✅ GOOD: Always await async calls
async def good_await():
    result = await some_async_function()
    return result
```

**Debugging Tools:**
```python
# Enable async debugging
import asyncio
import warnings

# Enable asyncio debug mode
asyncio.get_event_loop().set_debug(True)

# Detect unawaited coroutines
warnings.filterwarnings("always", category=RuntimeWarning, module="asyncio")

# Check for blocked event loop
import signal
import traceback

def debug_signal_handler(sig, frame):
    """Print stack trace on SIGUSR1"""
    print("=== STACK TRACE ===")
    traceback.print_stack(frame)
    print("==================")

signal.signal(signal.SIGUSR1, debug_signal_handler)
# Send signal: kill -USR1 <pid>
```

### Issue #3: Celery Task Failures & Retry Loops

**Symptoms:**
- Tasks stuck in retry loops
- High memory usage in workers
- Failed tasks not being logged

**Root Causes:**
```python
# ❌ BAD: No retry limits or error handling
@celery_app.task
def bad_task():
    # Might fail and retry forever
    risky_operation()

# ❌ BAD: No exception handling
@celery_app.task
def bad_task_2():
    api_call()  # If this fails, task crashes with no info
```

**✅ Solutions:**
```python
# ✅ GOOD: Proper task configuration
@celery_app.task(
    bind=True,
    autoretry_for=(requests.RequestException,),
    retry_kwargs={'max_retries': 3, 'countdown': 60},
    retry_backoff=True,
    retry_jitter=True
)
def good_task(self, data):
    try:
        result = risky_operation(data)
        return result
    except Exception as exc:
        logger.error(f"Task failed: {exc}")
        raise self.retry(exc=exc)

# ✅ GOOD: Task with progress tracking
@celery_app.task(bind=True)
def task_with_progress(self, items):
    total = len(items)
    for i, item in enumerate(items):
        try:
            process_item(item)
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={'current': i + 1, 'total': total}
            )
        except Exception as exc:
            logger.error(f"Failed to process item {item}: {exc}")
            continue
    
    return {'status': 'complete', 'processed': total}
```

**Debugging Tools:**
```python
# Monitor task status
from celery import current_app

def check_task_status(task_id):
    result = current_app.AsyncResult(task_id)
    print(f"State: {result.state}")
    print(f"Info: {result.info}")
    return result

# Check worker status
def check_workers():
    inspect = current_app.control.inspect()
    stats = inspect.stats()
    active = inspect.active()
    reserved = inspect.reserved()
    
    print(f"Worker stats: {stats}")
    print(f"Active tasks: {active}")
    print(f"Reserved tasks: {reserved}")

# Flower monitoring in development
# docker-compose up celery_flower
# Visit: http://localhost:5555
```

### Issue #4: Memory Leaks in Long-Running Processes

**Symptoms:**
- Gradually increasing memory usage
- Worker processes killed by OOM
- Slow performance over time

**Root Causes:**
```python
# ❌ BAD: Global state accumulation
CACHE = {}  # Global cache that grows forever

@celery_app.task
def bad_caching_task(data):
    CACHE[data['id']] = process_data(data)  # Never cleaned up
    return CACHE[data['id']]

# ❌ BAD: Circular references
class BadClass:
    def __init__(self):
        self.data = []
        self.parent = None
    
    def add_child(self, child):
        child.parent = self  # Circular reference
        self.data.append(child)
```

**✅ Solutions:**
```python
# ✅ GOOD: Bounded cache with cleanup
import weakref
from cachetools import TTLCache

# Use TTL cache instead of unbounded dict
cache = TTLCache(maxsize=1000, ttl=3600)

@celery_app.task
def good_caching_task(data):
    cache[data['id']] = process_data(data)
    return cache[data['id']]

# ✅ GOOD: Use weak references to break cycles
class GoodClass:
    def __init__(self):
        self.data = []
        self._parent = None
    
    @property
    def parent(self):
        return self._parent() if self._parent else None
    
    @parent.setter
    def parent(self, value):
        self._parent = weakref.ref(value) if value else None
```

**Debugging Tools:**
```python
# Memory profiling
import psutil
import gc
import objgraph

def memory_debug():
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
    
    # Show most common objects
    objgraph.show_most_common_types(limit=20)
    
    # Force garbage collection
    collected = gc.collect()
    print(f"Garbage collected: {collected} objects")

# Add to Celery tasks for monitoring
@celery_app.task
def monitored_task():
    import tracemalloc
    tracemalloc.start()
    
    # Your task logic here
    result = do_work()
    
    # Memory snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
    
    return result
```

### Issue #5: External API Rate Limiting & Failures

**Symptoms:**
- HTTP 429 (Too Many Requests) errors
- Tasks failing in batches
- External API timeouts

**Root Causes:**
```python
# ❌ BAD: No rate limiting or retry logic
async def bad_api_call():
    response = await httpx.get("https://api.example.com/data")
    return response.json()  # Fails on rate limit

# ❌ BAD: Batch operations without throttling
async def bad_batch_processing(items):
    tasks = [api_call(item) for item in items]
    results = await asyncio.gather(*tasks)  # Overwhelms API
    return results
```

**✅ Solutions:**
```python
# ✅ GOOD: Rate limiting with backoff
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class RateLimiter:
    def __init__(self, calls_per_second=10):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_called = 0
    
    async def acquire(self):
        elapsed = time.time() - self.last_called
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_called = time.time()

rate_limiter = RateLimiter(calls_per_second=5)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def good_api_call():
    await rate_limiter.acquire()
    
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        
        if response.status_code == 429:
            # Parse retry-after header if available
            retry_after = int(response.headers.get('retry-after', 60))
            await asyncio.sleep(retry_after)
            raise httpx.HTTPStatusError("Rate limited", request=None, response=response)
        
        response.raise_for_status()
        return response.json()

# ✅ GOOD: Throttled batch processing
async def good_batch_processing(items, concurrency=3):
    semaphore = asyncio.Semaphore(concurrency)
    
    async def throttled_call(item):
        async with semaphore:
            return await api_call(item)
    
    tasks = [throttled_call(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

## 🎯 Monitoring & Alerting Setup

### 1. Essential Metrics to Monitor

```python
# Custom metrics (add to your services)
from prometheus_client import Counter, Histogram, Gauge

# API metrics
api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration')

# Task metrics
task_duration = Histogram('celery_task_duration_seconds', 'Task duration', ['task_name'])
task_failures = Counter('celery_task_failures_total', 'Task failures', ['task_name', 'error_type'])

# Database metrics
db_connections = Gauge('db_connections_active', 'Active database connections')
db_query_duration = Histogram('db_query_duration_seconds', 'Database query duration')

# Memory metrics
memory_usage = Gauge('process_memory_usage_bytes', 'Process memory usage')
```

### 2. Health Checks

```python
# Comprehensive health check endpoint
@router.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Database check
    try:
        await execute_query("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Redis check
    try:
        import redis
        r = redis.from_url(settings.redis_url)
        r.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Celery check
    try:
        from src.tasks.celery_app import celery_app
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        if stats:
            health_status["checks"]["celery"] = "healthy"
        else:
            health_status["checks"]["celery"] = "no workers"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["celery"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return health_status
```

### 3. Alerting Rules (Prometheus)

```yaml
# monitoring/alert_rules.yml
groups:
  - name: ai_analysis_system
    rules:
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: DatabaseConnectionPoolExhausted
        expr: db_connections_active >= 18
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Database connection pool nearly exhausted"
          
      - alert: CeleryWorkerDown
        expr: up{job="celery"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Celery worker is down"
          
      - alert: HighMemoryUsage
        expr: process_memory_usage_bytes > 1000000000  # 1GB
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
```

## 🧪 Testing Strategies

### 1. Async Testing Setup

```python
# conftest.py
import pytest
import pytest_asyncio
from httpx import AsyncClient
from src.main import app
from src.database import get_db

@pytest_asyncio.fixture
async def test_db():
    # Setup test database
    pass

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

# Test async endpoints
@pytest.mark.asyncio
async def test_upload_document(client):
    response = await client.post("/api/v1/documents/upload", files={"file": ("test.txt", "content")})
    assert response.status_code == 201
```

### 2. Load Testing

```python
# load_test.py
import asyncio
import aiohttp
import time

async def load_test():
    connector = aiohttp.TCPConnector(limit=100)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        start_time = time.time()
        
        for i in range(1000):
            task = session.get(f"http://localhost:8000/api/v1/documents/")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful = sum(1 for r in responses if not isinstance(r, Exception))
        print(f"Completed {successful}/1000 requests in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(load_test())
```

## 🚀 Deployment Best Practices

### 1. Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Use strong `SECRET_KEY`
- [ ] Configure proper database connection pooling
- [ ] Set up log aggregation (ELK stack or similar)
- [ ] Configure monitoring and alerting
- [ ] Set up backup and recovery procedures
- [ ] Implement proper security headers and HTTPS
- [ ] Test disaster recovery procedures

### 2. Scaling Considerations

```yaml
# docker-compose.prod.yml scaling
version: '3.8'
services:
  app:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
  
  celery_worker:
    deploy:
      replicas: 5
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

This comprehensive guide should help you avoid the most common pitfalls and debug issues efficiently when they arise. The key is proactive monitoring, proper async patterns, and robust error handling from day one.
