@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ==========================================
echo  Whisper STT for RTX 5070 - Windows setup
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [x] Python not found.
    echo     Install Python 3.10-3.14 from python.org and enable "Add Python to PATH".
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment .venv
python -m venv .venv
if errorlevel 1 goto :err
call .venv\Scripts\activate.bat

echo [2/5] Upgrading pip
python -m pip install --upgrade pip

echo [3/5] Installing faster-whisper + yt-dlp
pip install -r requirements.txt
if errorlevel 1 goto :err

echo [4/5] Installing CUDA runtime DLLs (cuBLAS + cuDNN)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

if /i "%~1"=="--with-transformers" (
    echo [5/5] Installing torch cu128 + transformers ^(fallback engine^)
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 pip install torch
    pip install -r requirements-transformers.txt
) else (
    echo [5/5] Skipping transformers fallback ^(use: setup.bat --with-transformers^)
)

echo.
echo ==========================================
echo  Done.
echo  Run:  run.bat video.mp4 -o transcript.txt
echo ==========================================
pause
exit /b 0

:err
echo.
echo [x] Setup failed. See the error above.
pause
exit /b 1
