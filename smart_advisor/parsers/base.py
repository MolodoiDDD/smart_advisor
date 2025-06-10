from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)

class BaseParser(ABC):
    def __init__(self):
        self._clean_regex = re.compile(r'\s+')
        self._config = {}

    def clean_text(self, text: str) -> str:
        return self._clean_regex.sub(' ', text).strip()

    @abstractmethod
    def parse(self, file_path: Path) -> Optional[Dict]:
        pass

    @staticmethod
    @abstractmethod
    def is_supported(file_path: Path) -> bool:
        pass

    def validate_content(self, content: str) -> bool:
        if not content or len(content) < 50:
            logger.warning(f"Content too short: {len(content)} chars")
            return False
        return True