# syntax=docker/dockerfile:1.7
#
# Multi-stage build with two runnable targets:
#   docker build --target api -t rag-api .     (default, last stage)
#   docker build --target ui  -t rag-ui  .
#
ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# Stage 1: builder - installs deps into a venv and bakes the HF models in
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch first: the default wheel drags in ~2 GB of CUDA libraries we never use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The API image does not need Streamlit; strip it from the shared requirements file.
COPY requirements.txt /tmp/requirements.txt
RUN grep -viE '^streamlit' /tmp/requirements.txt > /tmp/api-requirements.txt \
    && pip install -r /tmp/api-requirements.txt

# Bake models into the image so containers start fast and work offline.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} RERANKER_MODEL=${RERANKER_MODEL}
RUN python -c "import os; from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer(os.environ['EMBEDDING_MODEL']); CrossEncoder(os.environ['RERANKER_MODEL'])"

# ---------------------------------------------------------------------------
# Stage 2 (target: ui) - tiny Streamlit image
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS ui

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install "streamlit>=1.38,<2" "requests>=2.32"
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app
COPY --chown=app:app app.py .
USER app
ENV API_URL=http://api:8000
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]

# ---------------------------------------------------------------------------
# Stage 3 (target: api, default) - runtime image
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS api

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    DATA_DIR=/data \
    PORT=8000

# libgomp1: OpenMP runtime needed by FAISS/torch on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data /app \
    && chown -R app:app /data /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf

WORKDIR /app
COPY --chown=app:app rag_pipeline.py main.py evaluate.py ./
COPY --chown=app:app static ./static

USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ONE worker: the FAISS/BM25 index lives in process memory.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
