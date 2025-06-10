import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Optional
from smart_advisor.parsers.base import BaseParser
from smart_advisor.config import settings
import logging

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    def __init__(self):
        super().__init__()
        self._config = settings.get_parser_config("pdf")

    def parse(self, file_path: Path) -> Optional[Dict]:
        try:
            doc = fitz.open(file_path)
            text = ""
            meta = {
                "source": str(file_path),
                "type": "pdf"
            }

            if self._config.get("extract_metadata", True):
                meta.update({
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "pages": len(doc)
                })

            for page in doc:
                text += page.get_text("text") + "\n"

            text = self.clean_text(text)

            if not self.validate_content(text):
                return None

            return {
                "content": text[:settings.MAX_TEXT_LENGTH],
                "metadata": meta
            }
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {str(e)}")
            return None
        finally:
            if 'doc' in locals():
                doc.close()

    @staticmethod
    def is_supported(file_path: Path) -> bool:
        return file_path.suffix.lower() == '.pdf'