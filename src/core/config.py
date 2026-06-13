# Centralizes all app settings — reads from .env and validates them at startup.

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Defines every configurable value in the app. Missing required fields crash early with a clear error."""

    # LLM
    openai_api_key: str
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "wealth_docs"

    # Ingestion
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k_results: int = 5

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache()
def get_settings() -> Settings:
    """Returns the Settings instance — cached after first call so .env is only read once."""
    return Settings()