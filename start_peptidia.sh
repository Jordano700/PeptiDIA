#!/bin/bash
echo "Starting PeptiDIA..."
cd "$(dirname "$0")"

# Kill any existing Streamlit processes to free up ports
echo "Checking for existing Streamlit processes..."
pkill -f "streamlit" 2>/dev/null || true
sleep 2

echo "Waiting for ports to be available..."
sleep 1

if [ -x "./peptidia_env/bin/python" ]; then
    PY="./peptidia_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

echo "Launching PeptiDIA on port 8501..."
"$PY" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8501
if [ $? -ne 0 ]; then
    echo "ERROR: PeptiDIA failed to start on port 8501"
    echo "Trying alternative port 8502..."
    "$PY" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8502
fi
