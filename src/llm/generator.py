# Orchestrates the full query pipeline — retrieve, redact, prompt, call LLM, return structured response.

from openai import OpenAI
from src.retrieval.hybrid import HybridRetriever
from src.compliance.redactor import ComplianceRedactor
from src.llm.prompt_manager import PromptManager
from src.core.models import QueryResponse, Citation
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class AnswerGenerator:
    """Runs the full RAG pipeline for a query — retrieval, compliance, prompting, and LLM call."""

    def __init__(self, prompt_version: str = "v1"):
        """Initializes all pipeline components — retriever, redactor, prompt manager, and OpenAI client."""
        settings = get_settings()

        self.retriever = HybridRetriever()
        self.redactor = ComplianceRedactor()
        self.prompt_manager = PromptManager(version=prompt_version)
        self.prompt_version = prompt_version

        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def answer(self, query: str) -> QueryResponse:
        """
        Runs the full pipeline for a single query.
        Steps: sanitize -> retrieve -> redact -> build prompt -> call LLM -> return structured response.
        """
        logger.info(f"Processing query: '{query[:80]}...'")

        # Step 1 — sanitize query for prompt injection before anything else
        safe_query = self.prompt_manager.sanitize_query(query)

        # Step 2 — retrieve relevant chunks using hybrid search
        chunks = self.retriever.search(safe_query)

        if not chunks:
            logger.warning("No chunks retrieved — returning no-context response")
            return self._empty_response(query)

        # Step 3 — redact PII from retrieved chunks before they enter the prompt
        clean_chunks, audit_log = self.redactor.redact_chunks(chunks)

        if audit_log:
            logger.info(f"Compliance: {len(audit_log)} PII items redacted before LLM call")

        # Step 4 — build the full prompt from versioned template
        system_prompt = self.prompt_manager.get_system_prompt()
        user_message = self.prompt_manager.build_user_message(
            query=safe_query,
            context_chunks=clean_chunks,
        )

        # Step 5 — call GPT-4o and get the answer
        raw_answer = self._call_llm(system_prompt, user_message)

        # Step 6 — build structured citations from the chunks used
        citations = self._build_citations(clean_chunks)

        logger.info(f"Query answered using {len(citations)} source(s)")

        return QueryResponse(
            answer=raw_answer,
            citations=citations,
            model_used=self.model,
            prompt_version=self.prompt_version,
        )

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Sends the prompt to GPT-4o and returns the text response."""
        response = self.openai.chat.completions.create(
            model=self.model,
            temperature=self.prompt_manager.get_temperature(),
            max_tokens=self.prompt_manager.get_max_tokens(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()

    def _build_citations(self, chunks: list[dict]) -> list[Citation]:
        """Converts retrieved chunks into Citation objects attached to the response."""
        citations = []
        seen = set()

        for chunk in chunks:
            # Deduplicate — same page from same document only cited once
            key = (chunk["source_file"], chunk["page_number"])
            if key in seen:
                continue
            seen.add(key)

            citations.append(Citation(
                document=chunk["source_file"],
                page_number=chunk["page_number"],
                chunk_text=chunk["text"][:200],  # preview of supporting text
            ))

        return citations

    def _empty_response(self, query: str) -> QueryResponse:
        """Returns a safe fallback response when no relevant chunks were retrieved."""
        return QueryResponse(
            answer="I don't have sufficient information in the provided documents to answer this question.",
            citations=[],
            model_used=self.model,
            prompt_version=self.prompt_version,
        )