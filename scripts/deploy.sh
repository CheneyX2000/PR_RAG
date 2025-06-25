#!/bin/bash
# scripts/deploy.sh 

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="./backups"
LOG_FILE="./deploy_$(date +%Y%m%d_%H%M%S).log"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file $ENV_FILE not found"
        print_info "Please copy .env.production to .env and configure it"
        exit 1
    fi
    
    # Check required environment variables
    local required_vars=("DB_PASSWORD" "OPENAI_API_KEY")
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$ENV_FILE" || grep -q "^${var}=CHANGE_THIS" "$ENV_FILE"; then
            print_error "Required variable $var is not set properly in $ENV_FILE"
            exit 1
        fi
    done
    
    print_success "Prerequisites check passed"
}

# Function to backup database
backup_database() {
    print_info "Creating database backup..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Check if postgres is running
    if docker compose ps postgres | grep -q "running"; then
        local backup_file="$BACKUP_DIR/ragdb_$(date +%Y%m%d_%H%M%S).sql"
        
        if docker compose exec -T postgres pg_dump -U "${DB_USER:-raguser}" "${DB_NAME:-ragdb}" > "$backup_file" 2>>"$LOG_FILE"; then
            print_success "Database backed up to $backup_file"
            
            # Compress backup
            gzip "$backup_file"
            print_info "Backup compressed to ${backup_file}.gz"
        else
            print_warning "Database backup failed, continuing anyway..."
        fi
    else
        print_warning "PostgreSQL is not running, skipping backup"
    fi
}

# Function to pull latest code
update_code() {
    print_info "Checking for code updates..."
    
    # Check if git is available
    if command -v git &> /dev/null && [ -d .git ]; then
        # Stash any local changes
        git stash push -m "Deployment stash $(date +%Y%m%d_%H%M%S)" >>"$LOG_FILE" 2>&1
        
        # Pull latest changes
        if git pull origin main >>"$LOG_FILE" 2>&1; then
            print_success "Code updated to latest version"
        else
            print_warning "Git pull failed, using existing code"
        fi
    else
        print_info "Git not available, skipping code update"
    fi
}

# Function to deploy services
deploy_services() {
    local profile=""
    
    # Parse deployment profile
    case "${1:-basic}" in
        "basic")
            print_info "Deploying basic services (API, PostgreSQL, Redis)"
            ;;
        "nginx")
            print_info "Deploying with Nginx proxy"
            profile="--profile with-nginx"
            ;;
        "monitoring")
            print_info "Deploying with monitoring stack"
            profile="--profile monitoring"
            ;;
        "full")
            print_info "Deploying full stack (Nginx + Monitoring)"
            profile="--profile with-nginx --profile monitoring"
            ;;
        *)
            print_error "Unknown deployment profile: $1"
            exit 1
            ;;
    esac
    
    # Pull latest images
    print_info "Pulling latest Docker images..."
    docker compose $profile pull >>"$LOG_FILE" 2>&1
    
    # Build custom images
    print_info "Building application image..."
    docker compose $profile build --no-cache api >>"$LOG_FILE" 2>&1
    
    # Start services
    print_info "Starting services..."
    if docker compose $profile up -d >>"$LOG_FILE" 2>&1; then
        print_success "Services started successfully"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Function to wait for services
wait_for_services() {
    print_info "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker compose exec -T api curl -f http://localhost:8000/api/v1/health >/dev/null 2>&1; then
            print_success "API is healthy"
            break
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Services failed to become healthy"
        docker compose logs --tail=50 api >>"$LOG_FILE"
        exit 1
    fi
}

# Function to run migrations
run_migrations() {
    print_info "Running database migrations..."
    
    if docker compose exec -T api alembic upgrade head >>"$LOG_FILE" 2>&1; then
        print_success "Migrations completed successfully"
    else
        print_warning "Migrations failed or not configured"
    fi
}

# Function to show deployment summary
show_summary() {
    print_info "Deployment Summary:"
    echo "===================="
    docker compose ps
    echo "===================="
    
    print_success "Deployment completed!"
    echo ""
    echo "Access points:"
    echo "- API: http://localhost:8000"
    echo "- API Docs: http://localhost:8000/docs"
    
    if docker compose ps | grep -q "nginx"; then
        echo "- HTTPS: https://localhost"
    fi
    
    if docker compose ps | grep -q "grafana"; then
        echo "- Grafana: http://localhost:3000"
        echo "- Prometheus: http://localhost:9090"
    fi
    
    echo ""
    echo "Logs: $LOG_FILE"
    echo "To view live logs: docker compose logs -f"
}

# Main deployment flow
main() {
    echo "RAG System Production Deployment"
    echo "================================"
    echo "Deployment started at $(date)" | tee "$LOG_FILE"
    echo ""
    
    # Check what kind of deployment
    DEPLOYMENT_TYPE="${1:-basic}"
    
    # Run deployment steps
    check_prerequisites
    backup_database
    update_code
    deploy_services "$DEPLOYMENT_TYPE"
    wait_for_services
    run_migrations
    show_summary
    
    echo ""
    echo "Deployment completed at $(date)" | tee -a "$LOG_FILE"
}

# Show usage if --help is passed
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [deployment-type]"
    echo ""
    echo "Deployment types:"
    echo "  basic      - API, PostgreSQL, Redis (default)"
    echo "  nginx      - Basic + Nginx proxy"
    echo "  monitoring - Basic + Prometheus/Grafana"
    echo "  full       - Everything (Nginx + Monitoring)"
    echo ""
    echo "Example:"
    echo "  $0 full"
    exit 0
fi

# Run main function
main "$@"