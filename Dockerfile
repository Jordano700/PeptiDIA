# PeptiDIA - Docker Container for 100% Reproducible Deployment
# Usage: docker build -t peptidia . && docker run -p 8501:8501 peptidia

FROM python:3.12.2-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker caching)
COPY requirements-tested.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements-tested.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Health check (curl not needed for basic setup)
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]