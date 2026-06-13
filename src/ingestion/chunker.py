# Splits page text into chunks using Hybrid strategies — recursive by default, semantic available for high-value docs.

import re
from enum import Enum
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from src.core.models import DocumentChunk
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies — recursive is fast and free, semantic is slower but meaning-aware."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class DocumentChunker:
    """Splits pages into coherent chunks using a pluggable strategy — defaults to recursive for cost efficiency."""

    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE):
        """Initializes the chosen chunking strategy with settings from config."""
        self.strategy = strategy
        settings = get_settings()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

        if strategy == ChunkingStrategy.RECURSIVE:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        elif strategy == ChunkingStrategy.SEMANTIC:
            self.splitter = SemanticChunker(
                embeddings=OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.openai_api_key,
                ),
                breakpoint_threshold_type="percentile",
            )

    def chunk_pages(
        self,
        pages: list[tuple[str, int]],
        source_file: str,
    ) -> list[DocumentChunk]:
        """Takes loader output and returns a flat list of DocumentChunk objects regardless of strategy used."""

        all_chunks = []

        # Robust filename sanitization
        clean_stem = Path(source_file).stem.lower().replace(" ", "_")
        clean_stem = re.sub(r'[^a-z0-9_]', '', clean_stem)

        for page_text, page_number in pages:
            # Skip completely empty pages to prevent splitter errors
            if not page_text.strip():
                continue
                
            # Defensive handling for semantic chunking on low-text pages
            try:
                page_chunks = self.splitter.split_text(page_text)
            except Exception as e:
                logger.warning(f"Strategy {self.strategy.value} failed on page {page_number}. Falling back to recursive. Error: {e}")
                fallback_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                page_chunks = fallback_splitter.split_text(page_text)

            for chunk_index, chunk_text in enumerate(page_chunks):
                chunk_id = f"{clean_stem}_p{page_number}_c{chunk_index}"

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    metadata={"strategy": self.strategy.value}, # Merge upstream dicts here if necessary
                )
                all_chunks.append(chunk)

        logger.info(
            f"'{source_file}' produced {len(all_chunks)} chunks "
            f"using {self.strategy.value} strategy"
        )
        return all_chunks