# Defines shared data models used across all modules — acts as the data contract between pipeline stages.

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone


class DocumentChunk(BaseModel):
    """Represents a single piece of text extracted and chunked from a source document."""

    chunk_id: str          # unique ID e.g. "apple_10k_p12_c3"
    text: str              # the actual chunk text content
    source_file: str       # original filename e.g. "apple_10k_2024.pdf"
    page_number: int       # page this chunk came from
    chunk_index: int       # position of this chunk within the page
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # when this chunk was processed and stored
    metadata: dict = Field(default_factory=dict) # any extra info we want to attach (e.g. section headers, keywords)


class Citation(BaseModel):
    """Represents a source reference attached to an LLM answer — tells the user where the answer came from."""

    document: str          # source filename
    page_number: int       # page number in that document
    chunk_text: str        # the exact excerpt supporting the answer
    ingested_at: Optional[datetime] = None


class QueryResponse(BaseModel):
    """The final structured response returned by the API — answer plus where it came from."""

    answer: str
    citations: list[Citation]
    model_used: str
    prompt_version: str