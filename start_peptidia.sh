#!/bin/bash
echo "🧬 Starting PeptiDIA..."
cd "$(dirname "$0")"

if [ -x "./peptidia_env/bin/python" ]; then
    PY="./peptidia_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

"$PY" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8501
