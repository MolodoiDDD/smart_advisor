from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

    def set_embedding(self, embedding: List[float]) -> None:
        self.embedding = np.array(embedding)

@dataclass
class SearchResult:
    document_id: str
    document: Document
    score: float
    text: str

@dataclass
class Query:
    text: str
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None

@dataclass
class Response:
    query: Query
    answer: str
    sources: List[SearchResult]
    confidence: float 