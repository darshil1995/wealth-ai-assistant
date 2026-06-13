wealth-ai-assistant/
│
├── src/
│ ├── core/
shared utilities
│ │ ├── config.py
# all settings from .env in one place
│ │ ├── logger.py
# structured logging
│ │ └── models.py
# shared Pydantic data models
│ │
│ ├── ingestion/
building today
│ │ ├── __init__.py
│ │ ├── loader.py
# reads PDFs, TXTs, future: URLs, Excel
│ │ ├── chunker.py
# splits docs into overlapping chunks
│ │ ├── embedder.py
# converts chunks → vectors
│ │ └── pipeline.py
# orchestrates loader→chunker→embedder
│ │
│ ├── retrieval/
module 3
│ │ ├── __init__.py
│ │ ├── vector_store.py
# ChromaDB read/write
│ │ ├── bm25_index.py
# keyword search index
│ │ └── hybrid.py
# merges both results (RRF)
│ │
│ ├── compliance/
module 4
│ │ ├── __init__.py
│ │ └── redactor.py
# PII detection + redaction gate
│ │
│ ├── llm/
module 5
│ │ ├── __init__.py
│ │ ├── prompt_manager.py
# loads versioned prompts from YAML
│ │ └── generator.py
# calls GPT-4o, returns answer + citations
│ │
│ └── api/
module 7
│ ├── __init__.py
│ ├── routes.py
# FastAPI endpoints
│ └── mcp_server.py
# MCP tool wrapper
│
├── prompts/
│ └── v1.yaml
# prompt template, versioned
├── sample_docs/
# your SEC PDFs go here
├── tests/
│ └── test_ingestion.py
building today
├── main.py
# app entry point
├── .env
├── requirements.txt
└── Dockerfile