# Orchestrates the full ingestion flow — load, chunk, embed, and store documents into ChromaDB.

import chromadb
from pathlib import Path
from openai import OpenAI
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker, ChunkingStrategy
from src.core.config import get_settings
from src.core.models import DocumentChunk
from src.core.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Public interface for ingestion — callers only need this class, loader and chunker are internal details."""

    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE):
        """Sets up all pipeline components — loader, chunker, OpenAI embedder, and ChromaDB collection."""
        settings = get_settings()

        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(strategy=strategy)

        # OpenAI client for generating embeddings
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.embedding_model

        # ChromaDB persists to disk — safe to restart without losing data
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_file(self, file_path: str) -> int:
        """Runs the full pipeline on a single file — returns the number of chunks stored."""
        file_name = Path(file_path).name
        logger.info(f"Starting ingestion: {file_name}")

        pages = self.loader.load(file_path)
        chunks = self.chunker.chunk_pages(pages, source_file=file_name)
        stored = self._embed_and_store(chunks)

        logger.info(f"Ingestion complete: {stored} chunks stored for '{file_name}'")
        return stored

    def ingest_directory(self, dir_path: str) -> dict[str, int]:
        """Ingests all supported files in a folder — returns a dict of filename to chunk count."""
        results = {}
        directory = Path(dir_path)

        for file_path in directory.iterdir():
            if file_path.suffix in DocumentLoader.SUPPORTED_EXTENSIONS:
                count = self.ingest_file(str(file_path))
                results[file_path.name] = count
            else:
                logger.info(f"  Skipping unsupported file: {file_path.name}")

        return results

    def _embed_and_store(self, chunks: list[DocumentChunk]) -> int:
        """Embeds all chunks in one API call then upserts into ChromaDB — upsert prevents duplicates on re-ingestion."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]

        logger.info(f"  Embedding {len(chunks)} chunks...")
        response = self.openai.embeddings.create(
            input=texts,
            model=self.embedding_model,
        )
        embeddings = [item.embedding for item in response.data]

        # Store source metadata alongside each chunk for citations later
        metadatas = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "strategy": c.metadata.get("strategy", "recursive"),
                "ingested_at": c.ingested_at.isoformat(),
            }
            for c in chunks
        ]

        # Upsert means re-ingesting the same file won't create duplicate chunks
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(chunks)