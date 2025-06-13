@echo off
REM install.bat - RAG System Installation Script for Windows
setlocal enabledelayedexpansion

REM Enable UTF-8 for better character support
chcp 65001 >nul 2>&1

REM Define colors using ANSI escape codes (Windows 10+)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "NC=[0m"

echo =============================================
echo RAG System Installation Script for Windows
echo =============================================
echo.

REM Function to print colored output
goto :start

:print_success
echo %GREEN%✓ %~1%NC%
goto :eof

:print_error
echo %RED%✗ %~1%NC%
goto :eof

:print_warning
echo %YELLOW%⚠ %~1%NC%
goto :eof

:start

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Python is not installed or not in PATH"
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

REM Get Python version and verify it's 3.9+
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if !MAJOR! LSS 3 (
    call :print_error "Python !PYTHON_VERSION! is installed, but Python 3.9+ is required"
    pause
    exit /b 1
)
if !MAJOR! EQU 3 if !MINOR! LSS 9 (
    call :print_error "Python !PYTHON_VERSION! is installed, but Python 3.9+ is required"
    pause
    exit /b 1
)
call :print_success "Python !PYTHON_VERSION! found"

REM Check Docker
echo Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker is not installed"
    echo Please install Docker Desktop from https://docs.docker.com/desktop/windows/install/
    pause
    exit /b 1
)

REM Check if Docker is running
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker is installed but not running"
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)
call :print_success "Docker found and running"

REM Check docker-compose (both variants)
echo Checking docker-compose...
docker-compose --version >nul 2>&1
if %errorlevel% equ 0 (
    set "DOCKER_COMPOSE=docker-compose"
    call :print_success "docker-compose found"
) else (
    docker compose version >nul 2>&1
    if %errorlevel% equ 0 (
        set "DOCKER_COMPOSE=docker compose"
        call :print_success "docker compose found"
    ) else (
        call :print_error "docker-compose is not installed"
        echo Please install docker-compose or update Docker Desktop
        pause
        exit /b 1
    )
)

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    call :print_warning "Virtual environment already exists"
    set /p recreate="Do you want to recreate it? (y/n): "
    if /i "!recreate!"=="y" (
        rmdir /s /q venv
        python -m venv venv
        if %errorlevel% neq 0 (
            call :print_error "Failed to create virtual environment"
            pause
            exit /b 1
        )
        call :print_success "Virtual environment recreated"
    )
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        call :print_error "Failed to create virtual environment"
        pause
        exit /b 1
    )
    call :print_success "Virtual environment created"
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    call :print_error "Failed to activate virtual environment"
    pause
    exit /b 1
)
call :print_success "Virtual environment activated"

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
call :print_success "pip upgraded"

REM Install dependencies
echo.
echo Installing dependencies...
pip install -e . >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Failed to install dependencies"
    echo Please check the error messages above
    pause
    exit /b 1
)
call :print_success "Dependencies installed"

REM Ask about dev dependencies
echo.
set /p install_dev="Install development dependencies? (y/n): "
if /i "%install_dev%"=="y" (
    pip install -e ".[dev]" >nul 2>&1
    if %errorlevel% equ 0 (
        call :print_success "Development dependencies installed"
    ) else (
        call :print_warning "Some development dependencies failed to install"
    )
)

REM Create .env file
echo.
if not exist .env (
    echo Creating .env file from template...
    if exist .env.example (
        copy .env.example .env >nul
        call :print_success ".env file created"
        call :print_warning "Please edit .env and add your API keys"
    ) else (
        call :print_warning ".env.example not found, creating basic .env"
        (
            echo # RAG System Configuration
            echo DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ragdb
            echo OPENAI_API_KEY=your-key-here
            echo REDIS_URL=redis://localhost:6379
            echo DEFAULT_EMBEDDING_MODEL=text-embedding-ada-002
            echo DEFAULT_LLM_MODEL=gpt-4o-mini
        ) > .env
        call :print_success ".env file created with defaults"
    )
) else (
    call :print_warning ".env file already exists"
)

REM Check .env configuration
echo.
echo Checking .env configuration...
findstr /C:"OPENAI_API_KEY=your-key-here" .env >nul 2>&1
if %errorlevel% equ 0 (
    call :print_warning "OPENAI_API_KEY is not set in .env"
)
findstr /C:"DATABASE_URL=" .env >nul 2>&1
if %errorlevel% neq 0 (
    call :print_warning "DATABASE_URL is not set in .env"
)

REM Start infrastructure
echo.
echo Starting infrastructure services...
%DOCKER_COMPOSE% -f docker-compose.dev.yml up -d >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Failed to start infrastructure"
    echo Please check Docker is running and try again
    pause
    exit /b 1
)
call :print_success "Infrastructure started"

REM Wait for PostgreSQL with better checking
echo Waiting for PostgreSQL to be ready...
set attempts=0
:wait_postgres
set /a attempts+=1
%DOCKER_COMPOSE% -f docker-compose.dev.yml exec -T postgres pg_isready -U postgres >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "PostgreSQL is ready"
    goto :postgres_ready
)
if !attempts! geq 30 (
    call :print_error "PostgreSQL failed to start in time"
    echo Please check the logs: %DOCKER_COMPOSE% -f docker-compose.dev.yml logs postgres
    pause
    exit /b 1
)
echo|set /p="."
timeout /t 1 /nobreak >nul
goto :wait_postgres

:postgres_ready

REM Initialize database
echo.
echo Initializing database...
python quickstart.py
if %errorlevel% neq 0 (
    call :print_warning "Database initialization had issues"
    echo Please check the logs and ensure your .env file is configured correctly
) else (
    call :print_success "Database initialized successfully"
)

REM Final instructions
echo.
echo =============================================
call :print_success "Installation complete!"
echo =============================================
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo    %YELLOW%notepad .env%NC%
echo.
echo 2. Activate virtual environment:
echo    %YELLOW%venv\Scripts\activate%NC%
echo.
echo 3. Start the API server:
echo    %YELLOW%uvicorn src.rag_system.main:app --reload%NC%
echo.
echo 4. Visit API documentation:
echo    %YELLOW%http://localhost:8000/docs%NC%
echo.
echo To stop infrastructure:
echo    %YELLOW%%DOCKER_COMPOSE% -f docker-compose.dev.yml down%NC%
echo.
echo To view logs:
echo    %YELLOW%%DOCKER_COMPOSE% -f docker-compose.dev.yml logs -f%NC%
echo.
pause