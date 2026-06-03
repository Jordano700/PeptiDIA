# PeptiDIA - runs out of the box with Python 3.12 and all dependencies bundled.
FROM python:3.12-slim

# OpenMP runtime needed by xgboost / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY config/requirements.txt config/requirements.txt
RUN pip install --no-cache-dir -r config/requirements.txt

# Copy the application (code, pre-trained model, demo data, assets, theme)
COPY . .

# PYTHONPATH=/app so the `from src.peptidia...` imports resolve regardless of how
# Streamlit is launched (the console script does not add the working dir to sys.path).
# GPU is auto-detected at runtime (XGBoost uses CUDA if the container has GPU access,
# otherwise falls back to CPU). To force CPU, set PEPTIDIA_FORCE_CPU=1.
ENV PYTHONPATH=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read().strip()==b'ok' else 1)" || exit 1

CMD ["streamlit", "run", "src/peptidia/web/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
