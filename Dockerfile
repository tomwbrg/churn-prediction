# syntax=docker/dockerfile:1

FROM python:3.12-slim

# WORKDIR must not be the root directory: Streamlit >= 1.10 refuses to run an
# app from / (FileNotFoundError, streamlit#5239).
WORKDIR /app

# libgomp1 is LightGBM's OpenMP runtime — without it `import lightgbm` fails at
# load time. curl serves the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so the pip layer stays cached when only the code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The code, the trained model and the stratified sample — everything the app
# needs to serve predictions without reaching for the 568 MB raw dataset.
COPY src/ ./src/
COPY app.py ./
COPY data/train_sample.parquet ./data/
COPY models/ ./models/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
