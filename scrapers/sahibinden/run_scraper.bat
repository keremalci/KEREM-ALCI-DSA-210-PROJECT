@echo off
TITLE Sahibinden Scraper Runner

echo ==========================================
echo      Sahibinden Scraper Installer/Runner
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b
)

echo [INFO] Python found. Checking dependencies...

REM Install dependencies
echo [INFO] Installing required libraries...
pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

echo [INFO] Dependencies installed successfully.

echo ==========================================
echo      Starting Scraper...
echo ==========================================
echo.
echo NOTE: A Chrome window will open.
echo 1. Log in to your account.
echo 2. Come back to this window and press ENTER.
echo.

python scraper_sahibinden_uc.py

echo.
echo [INFO] Script finished.
pause
    