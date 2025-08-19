#!/bin/bash
# PeptiDIA Streamlit Web Interface Launcher

echo "🧬 Starting PeptiDIA Web Interface..."

# Select Python executable: prefer local venv, else current environment (supports Conda)
if [ -x "./peptidia_env/bin/python" ]; then
	PYTHON="./peptidia_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON="python3"
else
	PYTHON="python"
fi

# Ensure dependencies are installed (use locked requirements if available)
if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import streamlit
PY
then
	echo "❌ Streamlit not found. Installing required packages into current environment..."
	"$PYTHON" -m pip install -r "requirements.txt"
fi

# Check if port 8501 is already in use
if lsof -i :8501 >/dev/null 2>&1; then
    echo "⚠️  Port 8501 is already in use!"
    echo "🔍 Checking if it's another Streamlit instance..."
    
    if pgrep -f "streamlit.*streamlit_app.py" >/dev/null; then
        echo "✅ PeptiDIA is already running at http://localhost:8501"
        echo "📱 Open your browser and navigate to: http://localhost:8501"
        exit 0
    else
        echo "❌ Another application is using port 8501"
        echo "💡 You can run with a different port:"
        echo "   $PYTHON -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8502"
        exit 1
    fi
fi

echo "🚀 Launching Streamlit app..."
echo "📱 Access the interface at: http://localhost:8501"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Run with selected Python
"$PYTHON" -m streamlit run src/peptidia/web/streamlit_app.py \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --logger.level error \
    --theme.primaryColor "#2E86AB" \
    --theme.backgroundColor "#FFFFFF" \
    --theme.secondaryBackgroundColor "#F0F2F6" \
    --theme.textColor "#262730"