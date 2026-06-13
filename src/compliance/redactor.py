# Scans retrieved chunks for PII and redacts it before any text reaches the LLM.

from presidio_analyzer import AnalyzerEngine, RecognizerResult, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from src.core.logger import get_logger

logger = get_logger(__name__)

# PII entity types we care about in financial documents
ENTITIES_TO_REDACT = [
    "PERSON",           # client names
    "EMAIL_ADDRESS",    # advisor or client emails
    "PHONE_NUMBER",     # contact numbers
    "LOCATION",         # home addresses
    "CA_SIN",           # Canadian Social Insurance Numbers
    "CREDIT_CARD",      # card numbers
    "IBAN_CODE",        # bank account numbers
    "US_SSN",           # US Social Security Numbers (for cross-border clients)
]


class ComplianceRedactor:
    """Detects and redacts PII from text chunks using Microsoft Presidio — runs fully locally with no API calls."""

    def __init__(self):
        """Initializes Presidio engines and registers a custom Canadian SIN recognizer."""
        logger.info("Loading Presidio compliance engine...")
        self.analyzer = AnalyzerEngine()

        # Add custom high-confidence SIN recognizer — Presidio's built-in CA_SIN misses some formats
        sin_recognizer = PatternRecognizer(
            supported_entity="CA_SIN",
            patterns=[
                Pattern(
                    name="ca_sin_dashes",
                    regex=r"\b\d{3}-\d{3}-\d{3}\b",
                    score=0.95,
                ),
                Pattern(
                    name="ca_sin_spaces",
                    regex=r"\b\d{3}\s\d{3}\s\d{3}\b",
                    score=0.95,
                ),
                Pattern(
                    name="ca_sin_plain",
                    regex=r"\b\d{9}\b",
                    score=0.6,   # lower confidence — 9 digits alone could be other things
                ),
            ]
        )
        self.analyzer.registry.add_recognizer(sin_recognizer)

        self.anonymizer = AnonymizerEngine()
        logger.info("  → Compliance engine ready")

    def redact(self, text: str) -> tuple[str, list[dict]]:
        """
        Scans text for PII and replaces each finding with a typed placeholder.
        Returns the cleaned text and a log of what was redacted.
        """
        # Step 1 — detect all PII entities in the text
        findings = self.analyzer.analyze(
            text=text,
            entities=ENTITIES_TO_REDACT,
            language="en",
        )

        if not findings:
            return text, []

        # Step 2 — replace each finding with a readable placeholder
        operators = self._build_operators(findings)
        redacted_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=findings,
            operators=operators,
        )

        # Step 3 — build an audit log of what was found and removed
        audit_log = self._build_audit_log(findings, text)

        logger.info(
            f"Redacted {len(findings)} PII item(s): "
            f"{[f['entity_type'] for f in audit_log]}"
        )

        return redacted_result.text, audit_log

    def redact_chunks(self, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
        """Redacts PII from a list of retrieved chunks — returns cleaned chunks and combined audit log."""
        cleaned_chunks = []
        full_audit_log = []

        for chunk in chunks:
            cleaned_text, audit_entries = self.redact(chunk["text"])

            # Replace the original text with the redacted version
            cleaned_chunk = chunk.copy()
            cleaned_chunk["text"] = cleaned_text
            cleaned_chunk["was_redacted"] = len(audit_entries) > 0

            cleaned_chunks.append(cleaned_chunk)
            full_audit_log.extend(audit_entries)

        return cleaned_chunks, full_audit_log

    def _build_operators(
        self, findings: list[RecognizerResult]
    ) -> dict[str, OperatorConfig]:
        """Builds a redaction operator per entity type — each type gets its own readable placeholder."""
        operators = {}

        for finding in findings:
            entity_type = finding.entity_type
            if entity_type not in operators:
                # Replace with a typed tag so the LLM knows something was removed
                operators[entity_type] = OperatorConfig(
                    "replace",
                    {"new_value": f"[REDACTED-{entity_type}]"},
                )

        return operators

    def _build_audit_log(
        self, findings: list[RecognizerResult], original_text: str
    ) -> list[dict]:
        """Records what was found and where — never logs the actual PII value, only its type and position."""
        return [
            {
                "entity_type": finding.entity_type,
                "start": finding.start,
                "end": finding.end,
                "confidence": round(finding.score, 3),
                # We log position only — never the actual PII value itself
            }
            for finding in findings
        ]