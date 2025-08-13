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
    pip install -r requirements_streamlit.txt
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