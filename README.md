# Wealth Management Gen AI Assistant

A production-grade Gen AI assistant for wealth advisors that answers questions over financial documents using a RAG pipeline with PII redaction, prompt injection protection, versioned prompts, structured citations, a REST API, and an MCP server layer.

Designed for financial institutions where auditability, compliance, and accuracy are non-negotiable.

---

## What It Does

A wealth advisor types a question like _"What is the risk profile of the Vanguard S&P 500 ETF?"_ and the system answers using the actual prospectus, not the LLM's training data. Every answer includes citations showing the exact document and page number that supported it.

---

## Key Features

- **Hybrid Retrieval** - Vector search (ChromaDB) + BM25 keyword search merged with Reciprocal Rank Fusion
- **PII Redaction** - Compliance gate using Microsoft Presidio strips names, SINs, emails, and phone numbers before any text reaches the LLM
- **Prompt Injection Protection** - Defends against both direct (query) and indirect (document) injection attacks
- **Versioned Prompts** - Prompts stored as YAML files outside the codebase - auditable, rollback-ready
- **Structured Citations** - Every answer includes source document, page number, and supporting excerpt
- **MCP Server** - Exposes the pipeline as a tool for AI agents via Anthropic's Model Context Protocol
- **REST API** - FastAPI with `/query`, `/ingest`, and `/documents` endpoints

---

## Tech Stack

| Technology | Purpose |
|---|---|
| PyMuPDF (fitz) | PDF loading - best for complex financial document layouts |
| ChromaDB | Local vector store with cosine similarity |
| BM25Okapi | Keyword search - runs locally, no API cost |
| Reciprocal Rank Fusion | Merges vector and BM25 results by rank position |
| Microsoft Presidio | PII detection and redaction - runs fully locally |
| OpenAI GPT-4o | Answer generation at temperature 0.0 |
| OpenAI text-embedding-3-small | Chunk and query embeddings |
| FastAPI + uvicorn | REST API server |
| MCP | Agent tool interface (Anthropic Model Context Protocol) |
| Pydantic | Data contracts between pipeline modules |
| LangChain text splitters | Recursive character chunking |
| pytest | Automated test suite |

---

## Project Structure

```
wealth-ai-assistant/
│
├── src/
│   ├── core/
│   │   ├── config.py           # Centralized settings from .env
│   │   ├── logger.py           # Structured logging across all modules
│   │   └── models.py           # Shared Pydantic data contracts
│   │
│   ├── ingestion/
│   │   ├── loader.py           # Reads PDFs and TXT files using PyMuPDF
│   │   ├── chunker.py          # Splits pages into overlapping chunks
│   │   └── pipeline.py         # Orchestrates load → chunk → embed → store
│   │
│   ├── retrieval/
│   │   ├── vector_store.py     # Semantic search via ChromaDB
│   │   ├── bm25_index.py       # Keyword search via BM25
│   │   └── hybrid.py           # Merges results using RRF
│   │
│   ├── compliance/
│   │   └── redactor.py         # PII detection and redaction gate
│   │
│   ├── llm/
│   │   ├── prompt_manager.py   # Versioned prompts + injection protection
│   │   └── generator.py        # Full pipeline orchestration + LLM call
│   │
│   └── api/
│       ├── routes.py           # FastAPI REST endpoints
│       └── mcp_server.py       # MCP tool server for AI agents
│
├── prompts/
│   └── v1.yaml                 # Versioned prompt template
│
├── sample_docs/                # Drop your PDF/TXT documents here
├── tests/
│   └── test_pipeline.py        # 20 automated tests
│
├── main.py                     # App entry point
├── conftest.py                 # pytest path configuration
├── pytest.ini                  # pytest settings
├── requirements.txt
├── .env                        # Your API keys (never commit)
├── .gitignore
└── Dockerfile
```

---

## Running from Scratch

### Prerequisites

- Python 3.11+ (for local run) OR Docker (for containerized run)
- An OpenAI API key — get one at [platform.openai.com](https://platform.openai.com)
- Load at least $5 credit — the entire project costs ~$1–3 in API calls

---

### Step 1 - Clone and create virtual environment

```powershell
git clone <your-repo-url>
cd wealth-ai-assistant
python -m venv venv
venv\Scripts\activate
```

On Mac/Linux:
```bash
source venv/bin/activate
```

---

### Step 2 - Install dependencies

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

> `en_core_web_lg` is the spaCy model used by Presidio for named entity recognition. It takes a few minutes to download.

---

### Step 3 - Create your `.env` file

Create a file called `.env` in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

All other settings have sensible defaults defined in `src/core/config.py`.

---

### Step 4 - Add financial documents

Drop PDF or TXT files into the `sample_docs/` folder. Free sources:

- **Apple 10-K** - go to [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar), search `AAPL`, select Annual Filings, download any 10-K PDF
- **Vanguard prospectus** - search `Vanguard` on EDGAR and download any fund prospectus

---

### Step 5 - Ingest documents

```powershell
python -c "from src.ingestion.pipeline import IngestionPipeline; p = IngestionPipeline(); print(p.ingest_directory('./sample_docs'))"
```

This runs the full ingestion pipeline - loads PDFs, chunks them, embeds with OpenAI, and stores in ChromaDB. A `chroma_db/` folder appears in your project root once complete.

---

### Step 6 - Run tests

```powershell
pytest tests/test_pipeline.py -v
```

Expected output:
```
20 passed, 2 warnings in ~11s
```

---

### Step 7 - Start the API server

```powershell
python main.py
```

Server starts at `http://localhost:8000`

---

### Step 8 - Open interactive API docs

Go to `http://localhost:8000/docs` in your browser. FastAPI generates interactive documentation automatically - you can test all endpoints directly from the browser.

---

### Step 9 - Ask a question

**Using curl (PowerShell):**
```powershell
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"question": "What are the annual total returns of the Vanguard fund?"}'
```

**Example response:**
```json
{
  "answer": "Based on the provided documents, the Vanguard Extended Market Index Fund Institutional Shares returned 11.02% after taxes on distributions...",
  "citations": [
    {
      "document": "SPI 856 Vanguard Extended Market Index Fund.pdf",
      "page_number": 5,
      "preview": "Annual Total Returns - Vanguard Extended Market Index Fund..."
    }
  ],
  "metadata": {
    "model_used": "gpt-4o",
    "prompt_version": "v1",
    "citation_count": 1
  }
}
```

---

### Step 10 - Ingest new documents via API (optional)

```powershell
curl -X POST http://localhost:8000/ingest `
  -H "Content-Type: application/json" `
  -d '{"path": "./sample_docs/new_document.pdf"}'
```

---

### Step 11 - Run the MCP server (optional, for agent use)

```powershell
python src/api/mcp_server.py
```

The MCP server exposes two tools to AI agents:
- `query_documents` - answers financial questions using the RAG pipeline
- `list_documents` - lists all ingested documents in the knowledge base

---

## Running with Docker

The easiest way to run the project — no Python setup needed.

### Step 1 — Build and start

```powershell
docker-compose up --build
```

First run takes ~5 minutes to install dependencies and download the spaCy model. Every run after that starts in seconds using the cached image.

### Step 2 — Ingest your documents

With the container running, open a second terminal:

```powershell
docker-compose exec wealth-ai-assistant python -c \
  "from src.ingestion.pipeline import IngestionPipeline; p = IngestionPipeline(); print(p.ingest_directory('./sample_docs'))"
```

### Step 3 — Visit the API docs

```
http://localhost:8000/docs
```

### Step 4 — Stop the container

```powershell
docker-compose down
```

> **Note:** Your ChromaDB data and ingested documents persist between restarts via Docker volumes — you don't need to re-ingest every time.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check - returns service status |
| POST | `/query` | Ask a question - returns answer with citations |
| POST | `/ingest` | Ingest a file or directory into ChromaDB |
| GET | `/documents` | List all ingested documents and chunk counts |

---

## Configuration

All settings are in `src/core/config.py` and can be overridden in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | LLM model for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for chunk and query embeddings |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores data on disk |
| `CHROMA_COLLECTION_NAME` | `wealth_docs` | ChromaDB collection name |
| `CHUNK_SIZE` | `500` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks to retrieve per query |

> **Note:** When running via Docker, `CHROMA_PERSIST_DIR` is automatically set to `/app/chroma_db` by `docker-compose.yml` — no manual change needed.

---

## Adding a New Document Type

The loader is designed to be extended. To add Word document support for example:

1. Install `python-docx`
2. Add `".docx"` to `SUPPORTED_EXTENSIONS` in `src/ingestion/loader.py`
3. Add a `_load_docx()` method
4. Add one `elif` line in `load()`

Nothing else in the project changes.

---

## Adding a New Prompt Version

1. Copy `prompts/v1.yaml` to `prompts/v2.yaml`
2. Edit the prompt content and update the `changelog` field
3. Switch to the new version:

```python
generator = AnswerGenerator(prompt_version="v2")
```

Or via the API:
```json
{"question": "What is the MER?", "prompt_version": "v2"}
```
---

## What to Build Next

- **Snowflake backend** - swap local ChromaDB for production-scale vector storage (retrieval module is already isolated - one file change)
- **Cross-encoder re-ranking** - add a second-pass re-ranker between retrieval and the LLM for higher accuracy on critical queries
- **Evaluation pipeline** - build a QA test set from documents and measure retrieval recall and answer accuracy automatically on every prompt version change
- **Authentication** - add API key auth to the FastAPI endpoints before any production deployment

---

## Author
Darshil Shah
