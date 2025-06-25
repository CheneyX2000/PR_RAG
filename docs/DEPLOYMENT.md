# docs/DEPLOYMENT.md
# Production Deployment Guide

This guide covers deploying the RAG System to production using Docker Compose.

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Domain name (for SSL)
- SSL certificates
- At least 8GB RAM
- 50GB+ available storage

## 🚀 Quick Start

### 1. Prepare Environment

```bash
# Clone the repository
git clone https://github.com/CheneyX2000/another_RAG.git
cd another_RAG

# Copy production environment template
cp .env.production .env

# Edit with your production values
nano .env
```

### 2. Generate Secure Passwords

```bash
# Generate database password
openssl rand -base64 32

# Generate Redis password
openssl rand -base64 32

# Generate API keys
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Basic Deployment

```bash
# Pull latest images
docker-compose pull

# Start services (without optional components)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 4. Full Deployment with Monitoring

```bash
# Start with all optional services
docker-compose --profile monitoring --profile with-nginx up -d
```

## 🔧 Configuration

### Environment Variables

Required variables in `.env`:

```bash
# Database (use strong passwords!)
DB_PASSWORD=your-strong-password-here

# API Keys
OPENAI_API_KEY=sk-...
API_KEY_REQUIRED=true
VALID_API_KEYS=your-api-key-1,your-api-key-2

# Domain Configuration
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
CORS_ORIGINS=https://yourdomain.com
```

### SSL Certificates

Place your SSL certificates in `nginx/ssl/`:

```bash
mkdir -p nginx/ssl
cp /path/to/cert.pem nginx/ssl/
cp /path/to/key.pem nginx/ssl/
chmod 600 nginx/ssl/key.pem
```

For testing, generate self-signed certificates:

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

## 📊 Monitoring

### Access Monitoring Services

- **Prometheus**: http://your-domain:9090
- **Grafana**: http://your-domain:3000
  - Default login: admin/admin (change immediately)
- **API Metrics**: http://your-domain/metrics (restricted)

### Import Grafana Dashboards

1. Log into Grafana
2. Go to Dashboards → Import
3. Upload dashboard JSON files from `monitoring/grafana/dashboards/`

## 🔐 Security Checklist

- [ ] Change all default passwords
- [ ] Enable API key authentication
- [ ] Configure firewall rules
- [ ] Set up SSL certificates
- [ ] Restrict database access
- [ ] Enable security headers
- [ ] Configure rate limiting
- [ ] Set up backup strategy
- [ ] Monitor logs regularly

## 🔄 Maintenance

### Database Backups

```bash
# Backup database
docker-compose exec postgres pg_dump -U raguser ragdb > backup_$(date +%Y%m%d).sql

# Restore database
docker-compose exec -T postgres psql -U raguser ragdb < backup.sql
```

### Updating the Application

```bash
# Stop services
docker-compose down

# Pull latest changes
git pull

# Rebuild API image
docker-compose build api

# Start services
docker-compose up -d

# Run migrations if needed
docker-compose exec api alembic upgrade head
```

### Scaling

To run multiple API instances:

```bash
# Scale to 3 instances
docker-compose up -d --scale api=3
```

Note: Requires a load balancer (nginx handles this automatically).

## 🚨 Troubleshooting

### Check Service Health

```bash
# Check all services
docker-compose ps

# Check specific service logs
docker-compose logs -f api
docker-compose logs -f postgres

# Check API health
curl http://localhost:8000/api/v1/health
```

### Common Issues

**Database Connection Failed**
```bash
# Check PostgreSQL is running
docker-compose exec postgres pg_isready

# Check credentials
docker-compose exec postgres psql -U raguser -d ragdb
```

**High Memory Usage**
```bash
# Check memory usage
docker stats

# Adjust limits in docker-compose.yml
# Under deploy.resources.limits
```

**SSL Certificate Issues**
```bash
# Check certificate validity
openssl x509 -in nginx/ssl/cert.pem -text -noout

# Check nginx configuration
docker-compose exec nginx nginx -t
```

## 📈 Performance Tuning

### PostgreSQL Optimization

Edit PostgreSQL settings in docker-compose.yml:

```yaml
environment:
  - POSTGRES_MAX_CONNECTIONS=200
  - POSTGRES_SHARED_BUFFERS=1GB
  - POSTGRES_EFFECTIVE_CACHE_SIZE=3GB
```

### Redis Optimization

Adjust memory limits based on your needs:

```yaml
command: >
  redis-server
  --maxmemory 2gb
  --maxmemory-policy allkeys-lru
```

### API Performance

Configure in `.env`:

```bash
MAX_WORKERS=8  # Based on CPU cores
CONNECTION_POOL_SIZE=50  # Based on concurrent users
```

## 🌐 Production Best Practices

1. **Use a CDN** for static assets
2. **Enable log aggregation** (ELK stack, CloudWatch, etc.)
3. **Set up alerts** for critical metrics
4. **Implement zero-downtime deployments**
5. **Regular security audits**
6. **Automated backups** with offsite storage
7. **Load testing** before major updates
8. **Document your deployment process**

## 📞 Support

For production support:

1. Check logs: `docker-compose logs -f`
2. Review monitoring dashboards
3. Check the [Issues](https://github.com/CheneyX2000/another_RAG/issues) page
4. Contact support with logs and configuration details