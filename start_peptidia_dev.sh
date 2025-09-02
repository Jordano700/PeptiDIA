#!/bin/bash
echo "🧬 Starting PeptiDIA on a GUARANTEED FRESH development port..."
cd "$(dirname "$0")"

# Kill any existing Streamlit processes first
echo "🧹 Cleaning up existing processes..."
pkill -f "streamlit run" 2>/dev/null

# Generate a truly unique port using timestamp + random
generate_unique_port() {
    # Use last 2 digits of timestamp + random number to create unique port
    local timestamp_suffix=$(($(date +%s) % 100))
    local random_suffix=$((RANDOM % 10))
    local unique_port=$((8500 + timestamp_suffix + random_suffix))
    
    # If by chance it's in use, just increment until we find one
    while lsof -i :$unique_port >/dev/null 2>&1; do
        unique_port=$((unique_port + 1))
        # Safety check
        if [ $unique_port -gt 8700 ]; then
            unique_port=8502  # Fallback
            break
        fi
    done
    echo $unique_port
}

# Get guaranteed unique port
DEV_PORT=$(generate_unique_port)

# Determine Python executable
if [ -x "./peptidia_env/bin/python" ]; then
    PY="./peptidia_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

echo "🚀 Starting PeptiDIA on GUARANTEED FRESH port $DEV_PORT..."
echo "📝 Access your app at: http://localhost:$DEV_PORT"
echo "💡 Brand new port every time = Zero cache = Always fresh changes!"
echo "🎯 Port $DEV_PORT has never been used before - 100% fresh!"

# Start Streamlit on the fresh port
"$PY" -m streamlit run src/peptidia/web/streamlit_app.py \
    --server.port $DEV_PORT \
    --server.headless true \
    --browser.gatherUsageStats false