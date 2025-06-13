#!/bin/bash
# install.sh - RAG System Installation Script for Linux/macOS

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================="
echo "RAG System Installation Script for Linux/macOS"
echo "============================================="
echo ""

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    print_error "Python is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

# Verify Python version is 3.9+
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $PYTHON_VERSION is installed, but Python 3.9+ is required"
    exit 1
fi
print_success "Python $PYTHON_VERSION found"

# Check Docker
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker ps &> /dev/null; then
    print_error "Docker is not running or you don't have permissions"
    echo "Please start Docker or add your user to the docker group:"
    echo "  sudo usermod -aG docker $USER"
    echo "  Then log out and back in"
    exit 1
fi
print_success "Docker found and running"

# Check docker-compose
echo "Checking docker-compose..."
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    print_error "docker-compose is not installed"
    echo "Please install docker-compose from https://docs.docker.com/compose/install/"
    exit 1
fi
print_success "docker-compose found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment recreated"
    fi
else
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -e . > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Ask about dev dependencies
echo ""
read -p "Install development dependencies? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[dev]" > /dev/null 2>&1
    print_success "Development dependencies installed"
fi

# Create .env file
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    print_success ".env file created"
    print_warning "Please edit .env and add your API keys"
else
    print_warning ".env file already exists"
fi

# Check if .env has required keys
echo ""
echo "Checking .env configuration..."
if [ -f .env ]; then
    if grep -q "OPENAI_API_KEY=your-key-here" .env || grep -q "OPENAI_API_KEY=$" .env; then
        print_warning "OPENAI_API_KEY is not set in .env"
    fi
    if grep -q "DATABASE_URL=$" .env; then
        print_warning "DATABASE_URL is not set in .env"
    fi
fi

# Start infrastructure
echo ""
echo "Starting infrastructure services..."
$DOCKER_COMPOSE_CMD -f docker-compose.dev.yml up -d > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Infrastructure started"
else
    print_error "Failed to start infrastructure"
    exit 1
fi

# Wait for PostgreSQL
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if $DOCKER_COMPOSE_CMD -f docker-compose.dev.yml exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        print_success "PostgreSQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "PostgreSQL failed to start in time"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Initialize database
echo ""
echo "Initializing database..."
python quickstart.py
if [ $? -eq 0 ]; then
    print_success "Database initialized successfully"
else
    print_warning "Database initialization had issues"
    echo "Please check the logs and ensure your .env file is configured correctly"
fi

# Final instructions
echo ""
echo "============================================="
print_success "Installation complete!"
echo "============================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "   ${YELLOW}vim .env${NC}"
echo ""
echo "2. Activate virtual environment:"
echo "   ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "3. Start the API server:"
echo "   ${YELLOW}uvicorn src.rag_system.main:app --reload${NC}"
echo ""
echo "4. Visit API documentation:"
echo "   ${YELLOW}http://localhost:8000/docs${NC}"
echo ""
echo "To stop infrastructure:"
echo "   ${YELLOW}$DOCKER_COMPOSE_CMD -f docker-compose.dev.yml down${NC}"
echo ""
echo "To view logs:"
echo "   ${YELLOW}$DOCKER_COMPOSE_CMD -f docker-compose.dev.yml logs -f${NC}"
echo ""