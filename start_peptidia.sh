#!/bin/bash
echo "Starting PeptiDIA..."
cd "$(dirname "$0")"

# Simple cleanup of existing processes
echo "Checking for existing Streamlit processes..."
pkill -f "streamlit" 2>/dev/null || true
sleep 1

if [ -x "./peptidia_env/bin/python" ]; then
    PY="./peptidia_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

echo "Launching PeptiDIA..."
"$PY" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8501
if [ $? -ne 0 ]; then
    echo "ERROR: PeptiDIA failed to start on port 8501"
    echo "Trying alternative port 8502..."
    "$PY" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8502
fi
