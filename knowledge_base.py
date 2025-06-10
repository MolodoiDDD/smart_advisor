import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict
from datetime import datetime

class KnowledgeBase:
    def __init__(self, persist_directory: str = "chroma_db_storage"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, text: str, metadata: Dict):
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata]
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return documents
    
    def update_document(self, doc_id: str, text: str, metadata: Dict):
        self.collection.update(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata]
        )
    
    def delete_document(self, doc_id: str):
        self.collection.delete(ids=[doc_id]) 