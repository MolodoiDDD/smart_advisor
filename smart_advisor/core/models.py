from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    id: Optional[str] = None

    def set_embedding(self, embedding: np.ndarray) -> None:
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "id": self.id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        return cls(
            text=data["text"],
            metadata=data["metadata"],
            embedding=np.array(data["embedding"]) if data.get("embedding") is not None else None,
            id=data.get("id")
        )

class BertModel:

    
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        self.model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
        self.model.eval()
        
    def get_embedding(self, text: str) -> np.ndarray:

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            return embeddings[0]

@dataclass
class SearchResult:
    document: Document
    score: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "text": self.text
        }