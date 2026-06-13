# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the image small — avoids pulling in unnecessary OS packages
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential needed for some Python packages that compile C extensions (e.g. chromadb)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# All app files live under /app inside the container
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first — Docker caches this layer so pip only re-runs when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Download spaCy model ──────────────────────────────────────────────────────
# Required by Presidio for named entity recognition (PII detection)
RUN python -m spacy download en_core_web_lg

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY main.py .
COPY conftest.py .
COPY pytest.ini .

# ── Create directories ────────────────────────────────────────────────────────
# sample_docs is where PDFs are mounted at runtime — chroma_db persists vector store
RUN mkdir -p sample_docs chroma_db

# ── Environment defaults ──────────────────────────────────────────────────────
# These are safe non-secret defaults — OPENAI_API_KEY must be passed at runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    OPENAI_MODEL=gpt-4o \
    EMBEDDING_MODEL=text-embedding-3-small \
    CHUNK_SIZE=500 \
    CHUNK_OVERLAP=50 \
    TOP_K_RESULTS=5

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the FastAPI server ──────────────────────────────────────────────────
CMD ["python", "main.py"]
