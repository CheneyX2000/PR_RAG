@echo off
REM install.bat - RAG System Installation Script for Windows

echo RAG System Installation Script for Windows
echo =========================================

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)
echo Python found

REM Check Docker
echo Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed
    echo Please install Docker Desktop from https://docs.docker.com/desktop/windows/install/
    pause
    exit /b 1
)
echo Docker found

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -e .
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed

REM Ask about dev dependencies
set /p install_dev=Install development dependencies? (y/n): 
if /i "%install_dev%"=="y" (
    pip install -e ".[dev]"
    echo Development dependencies installed
)

REM Create .env file
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo .env file created
    echo Please edit .env and add your API keys
) else (
    echo .env file already exists
)

REM Start infrastructure
echo Starting infrastructure services...
docker-compose -f docker-compose.dev.yml up -d
if %errorlevel% neq 0 (
    echo Error: Failed to start infrastructure
    pause
    exit /b 1
)
echo Infrastructure started

REM Wait for PostgreSQL
echo Waiting for PostgreSQL to be ready...
timeout /t 10 /nobreak >nul

REM Initialize database
echo Initializing database...
python quickstart.py
if %errorlevel% neq 0 (
    echo Warning: Database initialization had issues
    echo Please check the logs
)

echo.
echo Installation complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Activate virtual environment: venv\Scripts\activate
echo 3. Start the API server: uvicorn src.rag_system.main:app --reload
echo 4. Visit http://localhost:8000/docs for API documentation
echo.
echo To stop infrastructure: docker-compose -f docker-compose.dev.yml down
echo.
pause