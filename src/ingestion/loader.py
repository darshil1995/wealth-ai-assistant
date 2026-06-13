# Reads files from disk and extracts raw text — supports PDF and TXT, easily extendable to new formats.

from pathlib import Path
import fitz  # pymupdf
from src.core.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Loads raw text from supported file types and returns it as a list of (text, page_number) tuples."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def load(self, file_path: str) -> list[tuple[str, int]]:
        """Main entry point — detects file type and routes to the correct loader method."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"Loading file: {path.name}")

        if path.suffix == ".pdf":
            return self._load_pdf(path)
        elif path.suffix == ".txt":
            return self._load_txt(path)

    def _load_pdf(self, path: Path) -> list[tuple[str, int]]:
        """Extracts clean text page by page using PyMuPDF — handles complex layouts and financial tables better than pypdf."""
        doc = fitz.open(str(path))
        pages = []

        for page_num, page in enumerate(doc, start=1):
            # "text" mode preserves layout spacing better than default
            text = page.get_text("text")
            if text and text.strip():
                pages.append((text.strip(), page_num))

        doc.close()
        logger.info(f"Extracted {len(pages)} pages from {path.name}")
        return pages

    def _load_txt(self, path: Path) -> list[tuple[str, int]]:
        """Reads a plain text file as a single page."""
        text = path.read_text(encoding="utf-8")
        logger.info(f"Extracted text from {path.name}")
        return [(text.strip(), 1)]