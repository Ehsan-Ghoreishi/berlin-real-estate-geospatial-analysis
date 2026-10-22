# Reproducible runtime for the pipeline:
#   docker build -t berlin-real-estate .
#   docker run --rm -v "$PWD/data/processed:/app/data/processed" -v "$PWD/maps:/app/maps" \
#       berlin-real-estate --stages transform load model
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# rasterio's wheel (via contextily, for map basemaps) links against libexpat
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# dependencies first so code changes don't invalidate this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY src/ src/
COPY data/raw/ data/raw/

RUN useradd --create-home appuser && mkdir -p data/processed maps \
    && chown -R appuser /app
USER appuser

ENTRYPOINT ["python", "main.py"]
