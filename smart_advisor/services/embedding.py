from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, EmbeddingFunction

from ..config.config import config

class EmbeddingService(EmbeddingFunction):
    
    def __init__(self):
        self.model = SentenceTransformer(
            "sberbank-ai/sbert_large_nlu_ru",
            device='cpu',
            cache_folder=str(config.MODELS_DIR)
        )
        
    def __call__(self, input: Documents) -> List[List[float]]:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()