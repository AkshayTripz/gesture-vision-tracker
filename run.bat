@echo off
setlocal

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo ============================================
echo   Checking for Python 3.10 launcher...
echo ============================================
py -3.10 --version
if errorlevel 1 (
    echo ERROR: Python 3.10 not found via py launcher.
    echo Install it from https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
)

if not exist venv (
    echo ============================================
    echo   Creating venv with Python 3.10...
    echo ============================================
    py -3.10 -m venv venv
)

echo ============================================
echo   Activating venv...
echo ============================================
call venv\Scripts\activate.bat

echo ============================================
echo   Verifying correct interpreter is active...
echo ============================================
where python
python --version

echo ============================================
echo   Installing dependencies...
echo ============================================
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ============================================
echo   Verifying mediapipe.solutions is importable...
echo ============================================
python -c "import mediapipe as mp; print('mediapipe version:', mp.__version__); print(mp.solutions.hands)"
if errorlevel 1 (
    echo ============================================
    echo   ERROR: mediapipe.solutions is broken in this environment.
    echo   Do NOT proceed - check requirements.txt version pin.
    echo ============================================
    pause
    exit /b 1
)

echo ============================================
echo   All checks passed. Running main.py...
echo ============================================
python main.py

pause
