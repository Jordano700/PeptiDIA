@echo off
title PeptiDIA
echo Starting PeptiDIA...
cd /d "%~dp0"

REM Kill any existing Streamlit processes to free up ports
echo Checking for existing Streamlit processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq streamlit*" 2>nul
taskkill /F /IM python.exe /FI "COMMANDLINE eq *streamlit*" 2>nul
timeout /t 2 /nobreak >nul

REM Wait a moment for ports to be freed
echo Waiting for ports to be available...
timeout /t 3 /nobreak >nul

if exist "peptidia_env\\Scripts\\python.exe" (
  set "PY=peptidia_env\\Scripts\\python.exe"
) else (
  set "PY=python"
)

echo Launching PeptiDIA on port 8501...
"%PY%" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8501
if errorlevel 1 (
  echo.
  echo ERROR: PeptiDIA failed to start on port 8501
  echo Trying alternative port 8502...
  "%PY%" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8502
  if errorlevel 1 (
    echo.
    echo ERROR: PeptiDIA failed to start on both ports
    echo TIP: Try closing all browser tabs and restarting
    pause
  )
)