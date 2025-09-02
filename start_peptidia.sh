#!/bin/bash
echo "🧬 Starting PeptiDIA..."
cd "$(dirname "$0")"

# Kill any existing processes on port 8501
echo "🧹 Cleaning up existing processes..."
pkill -f "streamlit run" 2>/dev/null
lsof -ti :8501 | xargs -r kill -9 2>/dev/null
sleep 1

# Determine Python executable
if [ -x "./peptidia_env/bin/python" ]; then
    PY="./peptidia_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

echo "🚀 Launching PeptiDIA on port 8501..."
echo "📝 Access your app at: http://localhost:8501"

"$PY" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8501
