# Merges vector and BM25 search results into one ranked list using Reciprocal Rank Fusion.

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)

# RRF constant — 60 is the standard value from the original RRF paper
RRF_K = 60


class HybridRetriever:
    """Combines semantic and keyword search results into a single ranked list using RRF."""

    def __init__(self):
        """Initializes both vector store and BM25 index — ready to search immediately."""
        settings = get_settings()
        self.top_k = settings.top_k_results
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Runs both searches in parallel, merges with RRF, and returns the top_k best chunks."""
        top_k = top_k or self.top_k

        logger.info(f"Hybrid search: '{query[:60]}...'")

        # Run both searches — fetch more than top_k so RRF has enough candidates to merge
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

        # Merge using RRF
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # Return only the top_k after fusion
        top_results = fused[:top_k]

        logger.info(
            f"Hybrid returned {len(top_results)} chunks "
            f"(from {len(vector_results)} vector + {len(bm25_results)} bm25 candidates)"
        )
        return top_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """Scores each chunk by its rank position in both lists — chunks strong in both rank highest."""

        # Map chunk_id to fusion score, accumulating from both lists
        fusion_scores: dict[str, float] = {}

        # Map chunk_id to chunk data so we can reconstruct results after scoring
        chunk_lookup: dict[str, dict] = {}

        # Score vector results — rank 0 is best so we use (rank + 1) to avoid division by zero
        for rank, chunk in enumerate(vector_results):
            cid = chunk["chunk_id"]
            fusion_scores[cid] =  fusion_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
            chunk_lookup[cid] = chunk

        # Score BM25 results — add to existing score if chunk already appeared in vector results
        for rank, chunk in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            fusion_scores[cid] = fusion_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
            chunk_lookup[cid] = chunk

        # Sort by fusion score descending — highest score wins
        ranked_ids = sorted(
            fusion_scores.keys(),
            key=lambda cid: fusion_scores[cid],
            reverse=True,
        )

        # Rebuild result list with fusion scores attached
        results = []
        for cid in ranked_ids:
            chunk = chunk_lookup[cid].copy()
            chunk["rrf_score"] = round(fusion_scores[cid], 6)
            chunk["search_type"] = "hybrid"
            results.append(chunk)

        return results

    def rebuild_bm25(self):
        """Rebuilds the BM25 index — call this after ingesting new documents."""
        self.bm25_index.rebuild()