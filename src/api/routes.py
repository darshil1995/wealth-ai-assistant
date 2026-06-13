# Defines FastAPI HTTP endpoints — exposes the RAG pipeline as a REST API.

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.llm.generator import AnswerGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.core.logger import get_logger

logger = get_logger(__name__)

# FastAPI app instance — this is what uvicorn serves
app = FastAPI(
    title="Wealth Management Gen AI Assistant",
    description="RAG-powered assistant for wealth advisors — answers questions over financial documents.",
    version="1.0.0",
)

# Allow requests from any origin — tighten this in production to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline components once at startup — not on every request
generator = AnswerGenerator(prompt_version="v1")
ingestion_pipeline = IngestionPipeline()


# --- Request and Response Models ---

class QueryRequest(BaseModel):
    """Incoming query payload from the API caller."""
    question: str = Field(..., min_length=5, max_length=1000)
    prompt_version: str = Field(default="v1")


class IngestRequest(BaseModel):
    """Incoming ingestion request — points to a file or directory to ingest."""
    path: str = Field(..., description="Path to a file or directory to ingest")


# --- Endpoints ---

@app.get("/health")
def health_check():
    """Returns service health status — used by load balancers and monitoring tools."""
    return {"status": "healthy", "service": "wealth-ai-assistant"}


@app.post("/query")
def query(request: QueryRequest):
    """
    Main endpoint — takes a financial question and returns an answer with citations.
    Runs the full pipeline: sanitize → retrieve → redact → prompt → LLM → respond.
    """
    logger.info(f"POST /query — question: '{request.question[:80]}'")

    try:
        response = generator.answer(request.question)

        return {
            "answer": response.answer,
            "citations": [
                {
                    "document": c.document,
                    "page_number": c.page_number,
                    "preview": c.chunk_text,
                }
                for c in response.citations
            ],
            "metadata": {
                "model_used": response.model_used,
                "prompt_version": response.prompt_version,
                "citation_count": len(response.citations),
            }
        }

    except ValueError as e:
        # Raised by sanitize_query when injection is detected
        logger.warning(f"Query rejected: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/ingest")
def ingest(request: IngestRequest):
    """
    Ingestion endpoint — loads, chunks, embeds and stores documents from a given path.
    Accepts a file path or directory path.
    """
    import os
    logger.info(f"POST /ingest — path: '{request.path}'")

    try:
        if not os.path.exists(request.path):
            raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")

        if os.path.isdir(request.path):
            results = ingestion_pipeline.ingest_directory(request.path)
            return {
                "status": "success",
                "files_ingested": len(results),
                "chunks_stored": sum(results.values()),
                "details": results,
            }
        else:
            count = ingestion_pipeline.ingest_file(request.path)
            filename = os.path.basename(request.path)
            return {
                "status": "success",
                "files_ingested": 1,
                "chunks_stored": count,
                "details": {filename: count},
            }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """Returns a summary of all documents currently ingested in ChromaDB."""
    try:
        collection = ingestion_pipeline.collection
        results = collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return {"documents": [], "total_chunks": 0}

        # Aggregate chunk counts per source file
        doc_summary = {}
        for metadata in results["metadatas"]:
            source = metadata.get("source_file", "unknown")
            doc_summary[source] = doc_summary.get(source, 0) + 1

        return {
            "documents": [
                {"filename": filename, "chunks": count}
                for filename, count in doc_summary.items()
            ],
            "total_chunks": sum(doc_summary.values()),
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))