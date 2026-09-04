# App image for the `app` compose profile (FastAPI + Streamlit).
# Note: pulls torch via docling -- this image is large; the default workflow
# runs api/ui on the host and only postgres+qdrant in compose.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt requirements-ml.txt ./
RUN pip install -r requirements-ml.txt

COPY pyproject.toml .
COPY src ./src
COPY config ./config
RUN pip install -e . --no-deps

EXPOSE 8000 8501
CMD ["uvicorn", "dip.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
