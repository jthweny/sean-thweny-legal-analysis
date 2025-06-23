# AI Legal Analysis System

A comprehensive, continuously learning legal analysis system built with FastAPI, PostgreSQL, Redis, and modern AI APIs. This system processes legal documents, builds knowledge graphs, and provides intelligent analysis using multiple AI models.

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd fastapi_ai_system_template

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

**Copy the environment template:**
```bash
cp .env.example .env
```

**Edit `.env` and add your API keys:**
```bash
# Required API Keys - Add your actual keys here:
OPENAI_API_KEY="sk-your-openai-key-here"
ANTHROPIC_API_KEY="your-anthropic-key-here"
GEMINI_API_KEY="your-gemini-key-here"
FIRECRAWL_API_KEY="your-firecrawl-key-here"

# Database (default works for Docker Compose)
DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/ai_analysis"

# Redis (default works for Docker Compose)
REDIS_URL="redis://localhost:6379/0"

# Application Settings
DEBUG=true
SECRET_KEY="your-secret-key-change-in-production"
```

### 3. Test Configuration

```bash
# Test that your API keys are loaded correctly
python test_config.py
```

You should see output like:
```
=== Configuration Status ===
App Name: AI Analysis System
Version: 1.0.0
Debug Mode: True

=== API Keys Status ===
OpenAI: ✓ Set (sk-proj1...)
Anthropic: ✓ Set (sk-ant-...)
Gemini: ✓ Set (AIzaSy...)
Firecrawl: ✓ Set (fc-...)
```

### 4. Start with Docker Compose

```bash
# Start all services (PostgreSQL, Redis, FastAPI, Celery)
docker-compose up -d

# Check logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### 5. Access the Application

- **FastAPI API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Monitoring (Grafana)**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090

## 📁 Project Structure

```
fastapi_ai_system_template/
├── .env                    # Environment variables (create from .env.example)
├── .env.example           # Environment template with all settings
├── docker-compose.yml     # Full development environment
├── Dockerfile             # Application container
├── requirements.txt       # Python dependencies
├── test_config.py         # Configuration testing script
├── main.py               # FastAPI application entry point
├── init.sql              # Database schema
├── src/                  # Source code
│   ├── config.py         # Configuration management
│   ├── api_clients.py    # AI API integrations
│   ├── models/           # Database models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   └── tasks/            # Celery background tasks
├── monitoring/           # Prometheus and Grafana configs
└── docs/                # Documentation
```

## 🔑 API Keys Configuration

This system uses multiple AI APIs for comprehensive analysis. You'll need API keys from:

### Required APIs

1. **OpenAI** (GPT-4, embeddings)
   - Get key: https://platform.openai.com/api-keys
   - Add to `.env`: `OPENAI_API_KEY="sk-..."`

2. **Anthropic Claude** (Claude 3.5 Sonnet)
   - Get key: https://console.anthropic.com/
   - Add to `.env`: `ANTHROPIC_API_KEY="sk-ant-..."`

3. **Google Gemini** (Gemini 1.5 Flash)
   - Get key: https://makersuite.google.com/app/apikey
   - Add to `.env`: `GEMINI_API_KEY="AIza..."`

4. **Firecrawl** (Web scraping)
   - Get key: https://firecrawl.dev/
   - Add to `.env`: `FIRECRAWL_API_KEY="fc-..."`

### Configuration Best Practices

- **Never commit `.env` files** - they contain sensitive API keys
- Use different API keys for development and production
- Set rate limits appropriately for your usage
- Monitor API usage to avoid unexpected costs

## 🛠️ Development

### Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start PostgreSQL and Redis (Docker)
docker-compose up -d postgres redis

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run Celery worker (separate terminal)
celery -A src.tasks.celery_app worker --loglevel=info
```

### Testing API Integration

```bash
# Test all API integrations
python -m src.api_clients

# Test specific configuration
python test_config.py

# Run application tests
pytest

# Check API endpoints
curl http://localhost:8000/health
```

### Environment-Specific Configuration

Create different environment files:

- `.env.local` - Local development overrides
- `.env.staging` - Staging environment
- `.env.production` - Production settings (never commit!)

## 🔄 System Architecture

### Continuous Learning Loop

```
Document Upload → Content Extraction → Baseline Comparison → 
AI Analysis → Knowledge Graph Update → Pattern Detection → 
Insight Generation → Report Creation
```

### Key Features

- **Multi-Model Analysis**: Uses GPT-4, Claude 3.5, and Gemini for comprehensive insights
- **Smart Relevancy Scoring**: Compares new documents against baseline knowledge
- **Background Processing**: Celery workers handle heavy analysis tasks
- **Knowledge Accumulation**: Persistent knowledge graph builds over time
- **Web Research**: Automated research using Firecrawl when knowledge gaps detected
- **Professional Reports**: Generate comprehensive legal analysis documents

## 📊 Monitoring and Logging

### Built-in Monitoring

- **Health Checks**: `/health` endpoint for all services
- **Metrics**: Prometheus metrics at `/metrics`
- **Logging**: Structured JSON logging
- **Error Tracking**: Sentry integration (configure `SENTRY_DSN`)

### Grafana Dashboard

Access at http://localhost:3000 with admin/admin:

- System performance metrics
- API usage statistics
- Background task monitoring
- Error rates and latency

## 🚀 Production Deployment

### Environment Variables for Production

```bash
# Production settings in .env
DEBUG=false
LOG_LEVEL=WARNING
SECRET_KEY="your-secure-256-bit-secret"
ALLOWED_HOSTS="your-domain.com"
DATABASE_URL="postgresql+asyncpg://user:pass@prod-db:5432/ai_analysis"

# API keys (use secure secret management)
OPENAI_API_KEY="..."
ANTHROPIC_API_KEY="..."
# ... etc
```

### Docker Production Build

```bash
# Build production image
docker build -t ai-analysis-system .

# Run with production settings
docker run -d \
  --env-file .env.production \
  -p 8000:8000 \
  ai-analysis-system
```

### Scaling

- **Horizontal scaling**: Multiple FastAPI instances behind load balancer
- **Worker scaling**: Scale Celery workers based on queue size
- **Database**: Use read replicas for complex queries
- **Caching**: Redis for frequently accessed data

## 🔧 Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   # Check if .env file exists and has correct format
   cat .env | grep API_KEY
   
   # Test configuration
   python test_config.py
   ```

2. **Database Connection Error**
   ```bash
   # Check if PostgreSQL is running
   docker-compose ps postgres
   
   # Check connection
   docker-compose exec postgres psql -U postgres -d ai_analysis -c "SELECT 1;"
   ```

3. **Redis Connection Error**
   ```bash
   # Check if Redis is running
   docker-compose ps redis
   
   # Test connection
   docker-compose exec redis redis-cli ping
   ```

4. **Celery Worker Not Starting**
   ```bash
   # Check worker logs
   docker-compose logs celery_worker
   
   # Test Celery connection
   celery -A src.tasks.celery_app inspect ping
   ```

### Debug Mode

Enable debug mode in `.env`:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

This provides:
- Detailed logging
- Configuration status on startup
- FastAPI auto-reload
- Detailed error messages

## 📚 API Documentation

With the system running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Key Endpoints

- `POST /documents/upload` - Upload documents for analysis
- `GET /documents/{id}/analysis` - Get analysis results
- `GET /knowledge/insights` - Get accumulated insights
- `POST /reports/generate` - Generate comprehensive reports
- `GET /health` - Health check

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Copy `.env.example` to `.env` and configure
4. Run tests: `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push branch: `git push origin feature/amazing-feature`
7. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: See `docs/` directory
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions

---

**Ready to revolutionize legal analysis with AI?** 🚀

Follow the quick start guide above and you'll have a powerful, continuously learning legal analysis system running in minutes!
