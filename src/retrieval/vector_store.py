# Handles semantic search against ChromaDB — converts a query to a vector and finds the closest stored chunks.

import chromadb
from openai import OpenAI
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Wraps ChromaDB to provide semantic similarity search over ingested document chunks."""

    def __init__(self):
        """Connects to the existing ChromaDB collection and sets up the OpenAI embedder."""
        settings = get_settings()

        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.embedding_model
        self.top_k = settings.top_k_results

        # Connect to the same collection ingestion wrote to
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Embeds the query and returns the top_k most semantically similar chunks from ChromaDB."""
        top_k = top_k or self.top_k

        logger.info(f"Vector search: '{query[:60]}...'")

        # Convert the query text into a vector — same model used during ingestion
        query_embedding = self._embed_query(query)

        # Ask ChromaDB for the closest stored vectors
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Reformat into clean dicts for the hybrid layer to consume
        chunks = self._format_results(results)

        logger.info(f"  → Found {len(chunks)} chunks via vector search")
        return chunks

    def _embed_query(self, query: str) -> list[float]:
        """Converts query text into a vector using the same embedding model used at ingestion time."""
        response = self.openai.embeddings.create(
            input=[query],
            model=self.embedding_model,
        )
        return response.data[0].embedding

    def _format_results(self, raw_results: dict) -> list[dict]:
        """Converts ChromaDB's raw response format into a clean list of chunk dicts."""
        chunks = []

        # ChromaDB wraps results in an extra list because it supports batch queries. We always query one at a time so we take index [0].
        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "source_file": metadata.get("source_file", "unknown"),
                "page_number": metadata.get("page_number", 0),
                "score": round(1 - distance, 4),  # ChromaDB returns cosine distance (0 = identical, 2 = opposite). We convert it to a similarity score (1 = identical, 0 = opposite) by subtracting from 1. This makes scores intuitive — higher is better.
                "search_type": "vector",
            })

        return chunks