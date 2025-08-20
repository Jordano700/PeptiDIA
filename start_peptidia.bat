@echo off
title PeptiDIA
echo Starting PeptiDIA...
cd /d "%~dp0"

if exist "peptidia_env\\Scripts\\python.exe" (
  set "PY=peptidia_env\\Scripts\\python.exe"
) else (
  set "PY=python"
)

"%PY%" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8501