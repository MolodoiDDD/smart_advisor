from .config.config import config
from .models import Document, SearchResult
from .core.database import VectorDatabase
from .core.embeddings import EmbeddingService
from .core.file_loader import FileLoader

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

config.ensure_directories()

__all__ = [
    'config',
    'Document',
    'SearchResult',
    'VectorDatabase',
    'EmbeddingService',
    'FileLoader'
]

from pathlib import Path

__version__ = "0.1.0"

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "html").mkdir(exist_ok=True)
(DATA_DIR / "pdf").mkdir(exist_ok=True)

KB_STORAGE = Path(__file__).parent.parent / "knowledge_storage"
KB_STORAGE.mkdir(exist_ok=True)

CHROMA_STORAGE = Path(__file__).parent.parent / "chroma_db_storage"
CHROMA_STORAGE.mkdir(exist_ok=True)