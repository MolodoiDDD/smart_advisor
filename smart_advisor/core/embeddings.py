from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, EmbeddingFunction

from ..config.config import config

class EmbeddingService(EmbeddingFunction):

    
    def __init__(self):
        self.model = SentenceTransformer(
            config.EMBEDDING_CONFIG['model_name'],
            device=config.EMBEDDING_CONFIG['device'],
            cache_folder=config.EMBEDDING_CONFIG['cache_folder']
        )
        
    def __call__(self, input: Documents) -> List[List[float]]:

        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def generate_single(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist() 