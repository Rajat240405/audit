# Application image ONLY.
# Does not include Ollama, vLLM, CUDA, or Qwen weights.
# Inference is HTTP to a host process (Windows Ollama or HPC vLLM).
# Single Uvicorn worker — do not raise --workers.

# ── Stage 1: production frontend ──────────────────────────────────────────
FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: FastAPI + Hybrid RAG (CPU embeddings / FAISS) ────────────────
FROM python:3.12-slim-bookworm AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache \
    TRANSFORMERS_CACHE=/models/hf-cache \
    APP_DATA_DIR=/data \
    APP_INDEX_DIR=/storage/hybrid_rag \
    APP_MODEL_DIR=/models \
    APP_MODE=serve

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first so sentence-transformers does not pull a CUDA wheel.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && pip install \
        "httpx>=0.27.0" \
        "beautifulsoup4>=4.12.0" \
        "lxml>=5.0.0" \
        "pydantic>=2.7.0" \
        "pydantic-settings>=2.3.0" \
        "pyyaml>=6.0.0" \
        "rich>=13.7.0" \
        "tqdm>=4.66.0" \
        "click>=8.1.0" \
        "fastapi>=0.110.0" \
        "uvicorn>=0.28.0" \
        "orjson>=3.10.0" \
        "numpy>=1.24.0" \
        "pandas>=2.0.0" \
        "networkx>=3.0.0" \
        "faiss-cpu>=1.8.0" \
        "sentence-transformers>=3.0.0" \
        "rank-bm25>=0.2.2" \
        "pypdf>=5.0.0" \
        "python-docx>=1.1.0"

# Application source (no data / index / LLM weights).
COPY src /app/src
COPY config /app/config
COPY pyproject.toml /app/pyproject.toml

# Built workstation — FastAPI serves frontend/dist same-origin (/api relative).
COPY --from=frontend /frontend/dist /app/frontend/dist
# Backend loads this JSON at import time (PROJECT_ROOT/frontend/src/utils/...).
# It is not part of the Vite dist bundle — copy the file only, not frontend/src.
COPY frontend/src/utils/grounding_aliases.json /app/frontend/src/utils/grounding_aliases.json

# Empty mount points so the process never writes into CWD.
RUN mkdir -p /data /storage/hybrid_rag /models /models/hf-cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health/live || exit 1

# workers=1 is set inside start_server() — do not override with --workers.
# Phase-4 startup launcher: APP_MODE=serve starts WITHOUT ingest (default);
# APP_MODE=ingest runs the incremental ingestion pipeline first, then serves
# (python -m src.scripts.startup --help). Duplicate-process guard included.
CMD ["python", "-m", "src.scripts.startup"]
