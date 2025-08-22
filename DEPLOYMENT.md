# 🚀 SignaMentis Deployment Guide

## 📋 Overview

SignaMentis is a comprehensive AI-powered trading system with microservices architecture. This guide covers the complete deployment process using Docker and Docker Compose.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy  │    │   Prometheus    │    │     Grafana     │
│   (Port 80/443)│    │   (Port 9090)   │    │   (Port 3000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Trading System  │    │   News NLP      │    │     MLflow      │
│   (Port 8000)   │    │   (Port 8001)   │    │   (Port 5000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Redis       │    │    RabbitMQ     │    │     MongoDB     │
│   (Port 6379)   │    │   (Port 5672)   │    │   (Port 27017)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🐳 Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: Minimum 20GB free space
- **CPU**: 4+ cores recommended

### Software Installation

#### Docker Installation
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# macOS
brew install --cask docker

# Windows
# Download Docker Desktop from https://www.docker.com/products/docker-desktop
```

#### Docker Compose Installation
```bash
# Install Docker Compose plugin
sudo apt-get install docker-compose-plugin

# Verify installation
docker compose version
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-org/signa-mentis.git
cd signa-mentis
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. Deploy All Services
```bash
# Using deployment script (recommended)
python scripts/deploy.py deploy

# Or using Docker Compose directly
docker compose up -d
```

### 4. Verify Deployment
```bash
# Check service status
python scripts/docker_management.py status

# View logs
python scripts/docker_management.py logs trading_system
```

## 📚 Detailed Deployment

### Manual Deployment Steps

#### Step 1: Build Images
```bash
# Build all services
docker compose build

# Build specific service
docker compose build trading_system

# Build without cache
docker compose build --no-cache
```

#### Step 2: Start Infrastructure Services
```bash
# Start core infrastructure
docker compose up -d redis rabbitmq mongodb

# Wait for services to be ready
docker compose ps
```

#### Step 3: Start Application Services
```bash
# Start ML services
docker compose up -d mlflow minio

# Start monitoring
docker compose up -d prometheus grafana

# Start main application
docker compose up -d trading_system news_nlp
```

#### Step 4: Start Background Workers
```bash
# Start Celery workers
docker compose up -d celery_worker celery_beat

# Start monitoring tools
docker compose up -d flower
```

#### Step 5: Start Reverse Proxy
```bash
# Start Nginx
docker compose up -d nginx
```

### Service-Specific Configuration

#### Redis Configuration
```yaml
# docker-compose.yml
redis:
  environment:
    - REDIS_PASSWORD=your_secure_password
  volumes:
    - redis_data:/data
    - ./docker/redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
```

#### MongoDB Configuration
```yaml
mongodb:
  environment:
    - MONGO_INITDB_ROOT_USERNAME=admin
    - MONGO_INITDB_ROOT_PASSWORD=secure_password
    - MONGO_INITDB_DATABASE=signa_mentis
  volumes:
    - mongodb_data:/data/db
    - ./docker/mongo-init.js:/docker-entrypoint-initdb.d/mongo-init.js:ro
```

#### MLflow Configuration
```yaml
mlflow:
  environment:
    - MLFLOW_TRACKING_URI=http://localhost:5000
    - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    - AWS_ACCESS_KEY_ID=minioadmin
    - AWS_SECRET_ACCESS_KEY=minioadmin
```

## 🔧 Management Commands

### Using Docker Management Script
```bash
# Service management
python scripts/docker_management.py start
python scripts/docker_management.py stop
python scripts/docker_management.py restart

# View logs
python scripts/docker_management.py logs trading_system --follow
python scripts/docker_management.py logs news_nlp --tail 50

# Scale services
python scripts/docker_management.py scale trading_system 3

# Execute commands in containers
python scripts/docker_management.py exec trading_system "python -c 'print(\"Hello\")'"

# Backup and restore
python scripts/docker_management.py backup
python scripts/docker_management.py restore --backup-file backup_file.tar

# System metrics
python scripts/docker_management.py metrics
```

### Using Docker Compose Directly
```bash
# Service management
docker compose up -d
docker compose down
docker compose restart

# View logs
docker compose logs -f trading_system
docker compose logs --tail=100 news_nlp

# Scale services
docker compose up -d --scale trading_system=3

# Execute commands
docker compose exec trading_system bash
```

## 🧪 Testing and Validation

### Health Checks
```bash
# Check all endpoints
curl http://localhost:8000/health    # Trading System
curl http://localhost:8001/health    # News NLP
curl http://localhost:5000/health    # MLflow
curl http://localhost:3000/api/health # Grafana
curl http://localhost:9090/-/healthy # Prometheus
```

### Smoke Tests
```bash
# Run automated smoke tests
python scripts/deploy.py test

# Run specific test suites
python -m pytest tests/ -v
python tests/property_tests.py
python qa/gx/expectations.py
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

## 📊 Monitoring and Observability

### Prometheus Metrics
- **URL**: http://localhost:9090
- **Metrics Endpoints**:
  - Trading System: `/metrics`
  - News NLP: `/metrics`
  - MLflow: `/metrics`

### Grafana Dashboards
- **URL**: http://localhost:3000
- **Default Credentials**: admin / signa_mentis_2024
- **Pre-configured Dashboards**:
  - System Overview
  - Trading Performance
  - ML Model Metrics
  - Infrastructure Health

### Log Aggregation
```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f trading_system

# Search logs
docker compose logs | grep "ERROR"
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Generate self-signed certificates (development)
mkdir -p docker/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/signa_mentis.key \
  -out docker/nginx/ssl/signa_mentis.crt

# For production, use Let's Encrypt or commercial certificates
```

### Environment Variables
```bash
# .env file
REDIS_PASSWORD=your_secure_redis_password
MONGODB_PASSWORD=your_secure_mongodb_password
RABBITMQ_PASSWORD=your_secure_rabbitmq_password
VAULT_TOKEN=your_vault_token
API_SECRET_KEY=your_api_secret_key
```

### Network Security
```yaml
# docker-compose.yml
networks:
  default:
    name: signa_mentis_network
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 📈 Scaling and Performance

### Horizontal Scaling
```bash
# Scale trading system
docker compose up -d --scale trading_system=3

# Scale news NLP service
docker compose up -d --scale news_nlp=2

# Scale Celery workers
docker compose up -d --scale celery_worker=5
```

### Resource Limits
```yaml
# docker-compose.yml
trading_system:
  deploy:
    resources:
      limits:
        cpus: '2.0'
        memory: 4G
      reservations:
        cpus: '1.0'
        memory: 2G
```

### Load Balancing
```yaml
# docker-compose.yml
nginx:
  depends_on:
    - trading_system
  volumes:
    - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    - ./docker/nginx/upstream.conf:/etc/nginx/upstream.conf:ro
```

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker compose logs service_name

# Check resource usage
docker stats

# Verify dependencies
docker compose ps
```

#### Health Check Failures
```bash
# Check service health
docker compose ps

# Verify endpoints
curl -v http://localhost:port/health

# Check container status
docker inspect container_name
```

#### Database Connection Issues
```bash
# Test MongoDB connection
docker compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Test Redis connection
docker compose exec redis redis-cli ping

# Test RabbitMQ connection
docker compose exec rabbitmq rabbitmq-diagnostics ping
```

### Debug Mode
```bash
# Start services in debug mode
docker compose -f docker-compose.debug.yml up -d

# Enable verbose logging
docker compose logs -f --tail=1000
```

### Performance Issues
```bash
# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check network connectivity
docker network inspect signa_mentis_network

# Analyze container performance
docker compose exec service_name top
```

## 🔄 Backup and Recovery

### Automated Backups
```bash
# Create backup
python scripts/docker_management.py backup

# Schedule regular backups (cron)
0 2 * * * cd /path/to/signa-mentis && python scripts/docker_management.py backup
```

### Manual Backup
```bash
# Backup specific volumes
docker run --rm -v volume_name:/data -v $(pwd):/backup \
  alpine tar czf /backup/volume_name_$(date +%Y%m%d_%H%M%S).tar -C /data .

# Backup all data
docker run --rm -v /var/lib/docker/volumes:/volumes -v $(pwd):/backup \
  alpine tar czf /backup/all_volumes_$(date +%Y%m%d_%H%M%S).tar -C /volumes .
```

### Recovery Procedures
```bash
# Stop services
docker compose down

# Restore from backup
python scripts/docker_management.py restore --backup-file backup_file.tar

# Start services
docker compose up -d

# Verify recovery
python scripts/deploy.py test
```

## 🌍 Production Deployment

### Production Checklist
- [ ] SSL certificates configured
- [ ] Environment variables secured
- [ ] Resource limits set
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Security policies applied
- [ ] Load balancing configured
- [ ] CI/CD pipeline tested

### Production Commands
```bash
# Production deployment
python scripts/deploy.py deploy --environment production

# Production monitoring
python scripts/docker_management.py metrics

# Production backup
python scripts/docker_management.py backup --backup-dir /mnt/backups
```

### High Availability
```bash
# Multi-node deployment
docker swarm init
docker stack deploy -c docker-compose.swarm.yml signa_mentis

# Load balancer configuration
# Configure external load balancer (HAProxy, nginx, etc.)
```

## 📚 Additional Resources

### Documentation
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prometheus Documentation](https://prometheus.io/docs/)

### Support
- **Issues**: [GitHub Issues](https://github.com/your-org/signa-mentis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/signa-mentis/discussions)
- **Wiki**: [Project Wiki](https://github.com/your-org/signa-mentis/wiki)

### Community
- **Slack**: [SignaMentis Community](https://signa-mentis.slack.com)
- **Discord**: [Trading AI Community](https://discord.gg/trading-ai)
- **Meetups**: [Local AI Trading Meetups](https://meetup.com/ai-trading)

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Start everything
python scripts/deploy.py deploy

# Check status
python scripts/docker_management.py status

# View logs
python scripts/docker_management.py logs trading_system --follow

# Stop everything
python scripts/docker_management.py stop

# Clean up
python scripts/docker_management.py clean --volumes
```

### Service Ports
- **Trading System**: 8000
- **News NLP**: 8001
- **MLflow**: 5000
- **Grafana**: 3000
- **Prometheus**: 9090
- **Redis**: 6379
- **MongoDB**: 27017
- **RabbitMQ**: 5672, 15672 (Management)
- **MinIO**: 9000, 9001 (Console)

### Health Check URLs
- **Trading System**: http://localhost:8000/health
- **News NLP**: http://localhost:8001/health
- **MLflow**: http://localhost:5000/health
- **Grafana**: http://localhost:3000/api/health
- **Prometheus**: http://localhost:9090/-/healthy

---

**Happy Trading! 🚀📈**
