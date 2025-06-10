import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

from ..config.config import config
from ..models import Document, SearchResult
from .embeddings import EmbeddingService

class VectorDatabase:
    
    def __init__(self, embedding_service: EmbeddingService):
        self.client = chromadb.Client(Settings(
            persist_directory=str(config.DB_PATH),
            is_persistent=True
        ))
        
        self.embedding_service = embedding_service
        
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_service
        )
        
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
            
        self.collection.add(
            documents=[doc.text for doc in documents],
            ids=[doc.id for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
        
    def search(self, query: str, k: int = None) -> List[SearchResult]:

        results = self.collection.query(
            query_texts=[query],
            n_results=k or config.SEARCH_CONFIG['k']
        )
        
        return [
            SearchResult(
                document_id=id,
                document=Document(
                    id=id,
                    text=text,
                    metadata=metadata
                ),
                score=score,
                text=text
            )
            for id, score, metadata, text in zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0],
                results['documents'][0]
            )
            if score >= config.SEARCH_CONFIG['score_threshold']
        ]
        
    def clear(self) -> None:
        try:
            self.client.delete_collection("documents")
        except Exception as e:
            print(f"Ошибка при удалении коллекции: {str(e)}")
        
        self.collection = self.client.create_collection(
            name="documents",
            embedding_function=self.embedding_service,
            metadata={"hnsw:space": "cosine"}
        )