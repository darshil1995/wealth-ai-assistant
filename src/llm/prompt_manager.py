# Loads versioned prompt templates from YAML files — treats prompts as versioned config, not hardcoded strings.

import yaml
from pathlib import Path
import re
from src.core.logger import get_logger

logger = get_logger(__name__)

# Default prompts directory relative to project root
PROMPTS_DIR = Path("./prompts")


class PromptManager:
    """Loads and serves versioned prompt templates from YAML files in the prompts directory."""

    def __init__(self, version: str = "v1"):
        """Loads the specified prompt version from disk — fails fast if the version file doesn't exist."""
        self.version = version
        self.prompt_data = self._load_prompt(version)
        logger.info(f"Loaded prompt {version} — {self.prompt_data['changelog']}")

    def get_system_prompt(self) -> str:
        """Returns the system prompt string for this version."""
        return self.prompt_data["system_prompt"].strip()

    def get_answer_format(self) -> str:
        """Returns the answer format instructions appended to every user message."""
        return self.prompt_data["answer_format"].strip()

    def get_temperature(self) -> float:
        """Returns the temperature setting defined in this prompt version."""
        return float(self.prompt_data.get("temperature", 0.0))

    def get_max_tokens(self) -> int:
        """Returns the max token limit defined in this prompt version."""
        return int(self.prompt_data.get("max_tokens", 1000))

    def build_user_message(self, query: str, context_chunks: list[dict]) -> str:
        """Sanitizes the query for injection, then assembles the full user message with context and format instructions."""
        safe_query = self.sanitize_query(query)
        context_block = self._format_context(context_chunks)

        return (
            f"Context from financial documents:\n\n"
            f"{context_block}\n\n"
            f"Question: {safe_query}\n\n"
            f"{self.get_answer_format()}"
        )
            
    def sanitize_query(self, query: str) -> str:
        """Removes prompt injection patterns from user queries before they enter the prompt."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Flag common injection phrases — these have no legitimate place in a financial query
        injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
            r"you\s+are\s+now\s+a",
            r"forget\s+(everything|all|your)",
            r"new\s+instructions\s*:",
            r"system\s*prompt\s*:",
            r"reveal\s+(all|your|the)\s+(documents|chunks|context|instructions)",
            r"act\s+as\s+(if\s+you\s+are|a)",
            r"do\s+anything\s+now",
            r"jailbreak",
            r"pretend\s+you",
        ]

        query_lower = query.lower()
        for pattern in injection_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"Prompt injection attempt detected in query: '{query[:80]}'")
                raise ValueError(
                    "Query contains disallowed content and cannot be processed."
                )

        # Limit query length — very long queries are often injection attempts
        if len(query) > 1000:
            logger.warning(f"Query exceeds maximum length ({len(query)} chars) — truncating")
            query = query[:1000]

        return query.strip()

    def _sanitize_chunk_text(self, text: str) -> str:
        """Strips injection patterns from retrieved chunk text before it enters the context block."""
        injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
            r"you\s+are\s+now\s+a",
            r"forget\s+(everything|all|your)",
            r"new\s+instructions\s*:",
            r"system\s*prompt\s*:",
            r"act\s+as\s+(if\s+you\s+are|a)",
            r"do\s+anything\s+now",
            r"jailbreak",
            r"pretend\s+you",
        ]

        for pattern in injection_patterns:
            text = re.sub(pattern, "[REMOVED-INJECTION]", text, flags=re.IGNORECASE)

        return text

    def _format_context(self, chunks: list[dict]) -> str:
        """Formats retrieved chunks into a numbered context block — sanitizes each chunk for injection before use."""
        if not chunks:
            return "No relevant context found."

        formatted = []
        for i, chunk in enumerate(chunks, start=1):
            # Sanitize chunk text before it enters the prompt
            safe_text = self._sanitize_chunk_text(chunk["text"])

            source_line = (
                f"[Source {i}: {chunk['source_file']}, "
                f"Page {chunk['page_number']}]"
            )
            formatted.append(f"{source_line}\n{safe_text}")

        return "\n\n---\n\n".join(formatted)

    def _load_prompt(self, version: str) -> dict:
        """Reads and parses the YAML file for the given version — raises an error if not found."""
        prompt_path = PROMPTS_DIR / f"{version}.yaml"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt version '{version}' not found at {prompt_path}. "
                f"Available versions: {self._list_versions()}"
            )

        with open(prompt_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        logger.info(f"  → Prompt file loaded: {prompt_path}")
        return data

    def _list_versions(self) -> list[str]:
        """Returns all available prompt versions found in the prompts directory."""
        if not PROMPTS_DIR.exists():
            return []
        return [f.stem for f in PROMPTS_DIR.glob("*.yaml")]