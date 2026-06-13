# Builds a keyword search index over all stored chunks — finds exact term matches without any API calls.

import chromadb
from rank_bm25 import BM25Okapi
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class BM25Index:
    """Builds and queries a BM25 keyword index over all chunks currently stored in ChromaDB."""

    def __init__(self):
        """Loads all chunks from ChromaDB and builds the BM25 index in memory."""
        settings = get_settings()
        self.top_k = settings.top_k_results

        # Connect to the same ChromaDB collection
        chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Pull all stored chunks out of ChromaDB to build the index
        self.chunks = []
        self.index = None
        self._build_index()

    def _build_index(self):
        """Fetches all chunks from ChromaDB, tokenizes them, and builds the BM25 index."""
        logger.info("Building BM25 index from ChromaDB chunks...")

        # Fetch everything stored — no query, just get all documents
        results = self.collection.get(
            include=["documents", "metadatas"]
        )

        if not results["documents"]:
            logger.warning("No chunks found in ChromaDB — ingest documents first")
            return

        # Store raw chunk data for result lookup after search
        self.chunks = [
            {
                "chunk_id": chunk_id,
                "text": text,
                "source_file": metadata.get("source_file", "unknown"),
                "page_number": metadata.get("page_number", 0),
            }
            for chunk_id, text, metadata in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            )
        ]

        # BM25 needs tokenized text — simple whitespace split works well for financial text
        tokenized = [chunk["text"].lower().split() for chunk in self.chunks]
        self.index = BM25Okapi(tokenized)

        logger.info(f"BM25 index built over {len(self.chunks)} chunks")

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Scores all chunks against the query keywords and returns the top_k highest scoring chunks."""
        top_k = top_k or self.top_k

        if self.index is None:
            logger.warning("BM25 index is empty — returning no results")
            return []

        logger.info(f"BM25 search: '{query[:60]}...'")

        # Tokenize query the same way we tokenized the chunks
        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)

        # Get indices of top_k highest scores — argsort gives ascending, so reverse it
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            # Skip chunks with zero score — they share no keywords with the query
            if scores[idx] == 0:
                continue

            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "text": self.chunks[idx]["text"],
                "source_file": self.chunks[idx]["source_file"],
                "page_number": self.chunks[idx]["page_number"],
                "score": round(float(scores[idx]), 4),
                "search_type": "bm25",
            })

        logger.info(f"Found {len(results)} chunks via BM25 search")
        return results

    def rebuild(self):
        """Rebuilds the index from scratch — call this after ingesting new documents."""
        self._build_index()