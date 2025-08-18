#!/bin/bash

# 🧬 Interactive Peptide Validator - Launcher Script
# ================================================================================
# This script launches the Streamlit web interface for peptide validation

echo "🧬 Starting PeptiDIA..."
echo "🌐 The web interface will open automatically in your browser"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing required packages..."
    pip install -r requirements.txt
fi

# Check if port 8501 is already in use
if lsof -i :8501 >/dev/null 2>&1; then
    echo "⚠️  Port 8501 is already in use!"
    echo "🔍 Checking if it's another Streamlit instance..."
    
    if pgrep -f "streamlit.*streamlit_app.py" >/dev/null; then
        echo "✅ PeptiDIA is already running at http://localhost:8501"
        echo "📱 Open your browser and navigate to: http://localhost:8501"
        echo ""
        echo "💡 To restart the app:"
        echo "  1. Press Ctrl+C in the terminal where it's running"
        echo "  2. Or run: pkill -f streamlit"
        echo "  3. Then run this script again"
        exit 0
    else
        echo "❌ Another application is using port 8501"
        echo "💡 You can:"
        echo "  1. Stop the other application"
        echo "  2. Or modify this script to use a different port"
        exit 1
    fi
fi

# Set environment variables for optimal performance
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Launch Streamlit app
echo "🚀 Launching Streamlit app..."
echo "📱 Access the interface at: http://localhost:8501"
echo ""
echo "💡 Tips:"
echo "  - Use Ctrl+C to stop the server"
echo "  - Check the terminal for any error messages"
echo ""

# Run the Streamlit app
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --theme.primaryColor "#2E86AB" \
    --theme.backgroundColor "#FFFFFF" \
    --theme.secondaryBackgroundColor "#F0F2F6" \
    --theme.textColor "#262730"