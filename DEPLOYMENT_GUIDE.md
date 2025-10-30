# Equipment Verification System - Deployment Guide

This guide covers various deployment options for the Equipment Verification System, from local development to production environments.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Considerations](#production-considerations)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Git

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd equipment-verification

# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost
# Backend API: http://localhost:8000
```

### Option 2: Local Development

```bash
# Backend setup
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (in another terminal)
# Simply open frontend/index.html in a web browser
# Or serve with a simple HTTP server:
cd frontend
python -m http.server 8080
```

## 🛠️ Local Development

### Backend Setup

1. **Install Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

2. **Set Environment Variables**
```bash
export PYTHONPATH=/path/to/equipment-verification/backend
export MODEL_PATH=/path/to/your/model.pt  # Optional: custom model
```

3. **Run Development Server**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Test API**
```bash
curl http://localhost:8000/health
```

### Frontend Setup

1. **Serve Static Files**
```bash
cd frontend
python -m http.server 8080
```

2. **Access Application**
Open http://localhost:8080 in your browser

### Database Setup (Optional)

If you want to add persistent storage:

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb equipment_verification

# Update backend configuration
# Add database connection string to environment variables
```

## 🐳 Docker Deployment

### Single Container Deployment

1. **Build Backend Image**
```bash
cd backend
docker build -t equipment-verification-backend .
```

2. **Run Container**
```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  equipment-verification-backend
```

### Multi-Container Deployment

1. **Using Docker Compose**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

2. **Custom Configuration**
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  backend:
    environment:
      - MODEL_PATH=/app/models/custom_model.pt
    volumes:
      - ./custom_models:/app/models
```

### Production Docker Setup

1. **Multi-stage Dockerfile**
```dockerfile
# backend/Dockerfile.prod
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Production Docker Compose**
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    image: nginx:alpine
    restart: unless-stopped
    volumes:
      - ./frontend:/usr/share/nginx/html:ro
    ports:
      - "80:80"
      - "443:443"
```

## ☁️ Cloud Deployment

### AWS Deployment

1. **EC2 Instance**
```bash
# Launch EC2 instance (t3.medium or larger)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -a -G docker ec2-user

# Clone and deploy
git clone <repository-url>
cd equipment-verification
docker-compose up -d
```

2. **AWS ECS**
```yaml
# ecs-task-definition.json
{
  "family": "equipment-verification",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-account.dkr.ecr.region.amazonaws.com/equipment-verification:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PYTHONPATH",
          "value": "/app"
        }
      ]
    }
  ]
}
```

3. **AWS Lambda (Serverless)**
```python
# lambda_handler.py
import json
import base64
from app.models.detector import EquipmentDetector
from app.models.verifier import EquipmentVerifier

def lambda_handler(event, context):
    # Initialize models (consider using Lambda layers)
    detector = EquipmentDetector()
    verifier = EquipmentVerifier()
    
    # Process image
    image_data = base64.b64decode(event['body'])
    # ... detection logic
    
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

### Google Cloud Platform

1. **Cloud Run**
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/equipment-verification', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/equipment-verification']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'equipment-verification', '--image', 'gcr.io/$PROJECT_ID/equipment-verification', '--platform', 'managed', '--region', 'us-central1']
```

2. **Deploy to Cloud Run**
```bash
gcloud builds submit --config cloudbuild.yaml
```

### Azure Deployment

1. **Azure Container Instances**
```bash
# Create resource group
az group create --name equipment-verification --location eastus

# Deploy container
az container create \
  --resource-group equipment-verification \
  --name equipment-verification-app \
  --image your-registry.azurecr.io/equipment-verification:latest \
  --ports 8000 \
  --dns-name-label equipment-verification
```

2. **Azure App Service**
```yaml
# azure-pipelines.yml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: Docker@2
  inputs:
    containerRegistry: 'your-registry'
    repository: 'equipment-verification'
    command: 'buildAndPush'
    Dockerfile: 'backend/Dockerfile'
    tags: '$(Build.BuildId)'
```

## 🏭 Production Considerations

### Performance Optimization

1. **Model Optimization**
```python
# Use TensorRT for GPU acceleration
import tensorrt as trt

# Optimize model for inference
def optimize_model(model_path):
    # Convert to TensorRT
    # Apply quantization
    # Optimize for target hardware
    pass
```

2. **Caching Strategy**
```python
# Redis caching for model results
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_result(image_hash):
    return redis_client.get(f"result:{image_hash}")

def cache_result(image_hash, result):
    redis_client.setex(f"result:{image_hash}", 3600, result)
```

3. **Load Balancing**
```nginx
# nginx.conf
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Security Considerations

1. **API Security**
```python
# Add authentication middleware
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_token(token: str = Depends(security)):
    # Implement token verification
    if not is_valid_token(token.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token.credentials

# Apply to protected endpoints
@app.post("/detect")
async def detect_equipment(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    # ... detection logic
```

2. **Input Validation**
```python
# Validate uploaded files
def validate_image(file: UploadFile):
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    # Check file size
    if file.size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(400, "File too large")
    
    # Check for malicious content
    # ... additional validation
```

3. **Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/detect")
@limiter.limit("10/minute")
async def detect_equipment(request: Request, ...):
    # ... detection logic
```

### Monitoring and Logging

1. **Application Monitoring**
```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

2. **Structured Logging**
```python
import structlog

logger = structlog.get_logger()

@app.post("/detect")
async def detect_equipment(file: UploadFile = File(...)):
    logger.info("Processing image", 
                filename=file.filename, 
                size=file.size)
    
    try:
        # ... detection logic
        logger.info("Detection successful", 
                   detections_count=len(detections))
    except Exception as e:
        logger.error("Detection failed", error=str(e))
        raise
```

3. **Health Checks**
```python
@app.get("/health")
async def health_check():
    checks = {
        "database": check_database_connection(),
        "model": check_model_loaded(),
        "storage": check_storage_available()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": checks
        }
    )
```

## 🔧 Monitoring and Maintenance

### Log Management

1. **ELK Stack Setup**
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
```

2. **Grafana Dashboard**
```json
{
  "dashboard": {
    "title": "Equipment Verification Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(requests_total[5m])",
            "legendFormat": "{{endpoint}}"
          }
        ]
      }
    ]
  }
}
```

### Backup Strategy

1. **Model Backup**
```bash
#!/bin/bash
# backup_models.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/models/$DATE"

mkdir -p $BACKUP_DIR
cp -r models/* $BACKUP_DIR/
aws s3 sync $BACKUP_DIR s3://your-bucket/models/$DATE/
```

2. **Database Backup**
```bash
#!/bin/bash
# backup_database.sh
pg_dump equipment_verification > backup_$(date +%Y%m%d).sql
aws s3 cp backup_$(date +%Y%m%d).sql s3://your-bucket/database/
```

### Update Strategy

1. **Blue-Green Deployment**
```bash
# Deploy new version
docker-compose -f docker-compose.blue.yml up -d

# Test new version
curl http://new-version:8000/health

# Switch traffic
# Update load balancer configuration

# Shutdown old version
docker-compose -f docker-compose.green.yml down
```

2. **Rolling Updates**
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: equipment-verification
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      containers:
      - name: backend
        image: equipment-verification:latest
        ports:
        - containerPort: 8000
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
```bash
# Check model file exists
ls -la models/

# Verify model format
python -c "import torch; torch.load('models/best.pt')"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. **Memory Issues**
```bash
# Monitor memory usage
docker stats

# Increase memory limits
docker run --memory=4g equipment-verification-backend
```

3. **API Connection Issues**
```bash
# Test API connectivity
curl -v http://localhost:8000/health

# Check logs
docker-compose logs backend

# Verify port binding
netstat -tlnp | grep 8000
```

### Performance Debugging

1. **Profile Application**
```python
# Add profiling middleware
import cProfile
import pstats

@app.middleware("http")
async def profile_requests(request: Request, call_next):
    profiler = cProfile.Profile()
    profiler.enable()
    
    response = await call_next(request)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return response
```

2. **Monitor GPU Usage**
```bash
# NVIDIA GPU monitoring
nvidia-smi -l 1

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Log Analysis

1. **Error Pattern Detection**
```bash
# Find common errors
grep "ERROR" logs/app.log | sort | uniq -c | sort -nr

# Monitor error rate
grep "ERROR" logs/app.log | wc -l
```

2. **Performance Analysis**
```bash
# Find slow requests
grep "duration" logs/app.log | awk '{print $NF}' | sort -n | tail -10

# Monitor response times
grep "duration" logs/app.log | awk '{print $NF}' | awk '{sum+=$1; count++} END {print sum/count}'
```

## 📊 Scaling Considerations

### Horizontal Scaling

1. **Load Balancer Configuration**
```nginx
upstream backend {
    least_conn;
    server backend1:8000 weight=3;
    server backend2:8000 weight=3;
    server backend3:8000 weight=2;
}
```

2. **Database Scaling**
```yaml
# PostgreSQL cluster
services:
  postgres-master:
    image: postgres:13
    environment:
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: password

  postgres-slave:
    image: postgres:13
    environment:
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_MASTER_HOST: postgres-master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: password
```

### Vertical Scaling

1. **Resource Optimization**
```yaml
# Resource limits
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

2. **Model Optimization**
```python
# Model quantization
import torch.quantization as quantization

def quantize_model(model):
    model.eval()
    quantized_model = quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model
```

This deployment guide provides comprehensive instructions for deploying the Equipment Verification System in various environments, from local development to production cloud deployments. Choose the approach that best fits your requirements and infrastructure.

