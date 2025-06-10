from typing import List, Optional
from pathlib import Path

from ..models import Document
from ..config.config import config
from .file_loader import FileLoader
from .database import VectorDatabase

class DocumentProcessor:

    def __init__(self):
        self.file_loader = FileLoader()
        self.database = VectorDatabase()
        
    def get_documents(self) -> List[Document]:

        return self.file_loader.load_documents()
        
    def process_documents(self, documents_dir: Optional[Path] = None) -> None:

        documents = self.get_documents()
        

        self.database.clear()
        

        self.database.add_documents(documents)
        
    def search(self, query: str, k: Optional[int] = None) -> List[Document]:

        results = self.database.search(query, k)
        return results 