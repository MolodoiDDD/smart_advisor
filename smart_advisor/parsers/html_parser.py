from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, Optional
from smart_advisor.parsers.base import BaseParser
from smart_advisor.config import settings
import logging

logger = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    def __init__(self):
        super().__init__()
        self._config = settings.get_parser_config("html")

    def parse(self, file_path: Path) -> Optional[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

                for element in soup(self._config.get("exclude_tags", [])):
                    element.decompose()

                text = self.clean_text(soup.get_text())

                if not self.validate_content(text):
                    return None

                return {
                    "content": text[:settings.MAX_TEXT_LENGTH],
                    "metadata": {
                        "title": soup.title.string if soup.title else "",
                        "source": str(file_path),
                        "type": "html"
                    }
                }
        except Exception as e:
            logger.error(f"Failed to parse HTML {file_path}: {str(e)}")
            return None

    @staticmethod
    def is_supported(file_path: Path) -> bool:
        return file_path.suffix.lower() in ('.html', '.htm')