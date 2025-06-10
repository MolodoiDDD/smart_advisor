import json
import os
from pathlib import Path
from typing import Dict, Optional, List
from smart_advisor.config import settings, KNOWLEDGE_STORAGE_DIR
from smart_advisor.core.models import Document
import logging

logger = logging.getLogger(__name__)


class KnowledgeBase:

    def __init__(self):
        self.storage_dir = KNOWLEDGE_STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        try:
            settings.KB_STORAGE.mkdir(exist_ok=True, parents=True)
            logger.info(f"Knowledge base initialized at {settings.KB_STORAGE}")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {str(e)}")
            raise

    def save(self, document: Document) -> bool:

        try:
            file_path = os.path.join(self.storage_dir, f"{document.id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved document {document.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save document {document.id}: {str(e)}")
            return False

    def load(self, document_id: str) -> Optional[Document]:
        try:
            file_path = os.path.join(self.storage_dir, f"{document_id}.json")
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Document.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load document {document_id}: {str(e)}")
            return None

    def delete(self, document_id: str) -> bool:
        try:
            file_path = os.path.join(self.storage_dir, f"{document_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False

    def list_documents(self) -> list[str]:
        try:
            return [f.replace('.json', '') for f in os.listdir(self.storage_dir) 
                   if f.endswith('.json')]
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return []

    def get_all(self) -> List[Document]:
        docs = []
        for file_path in settings.KB_STORAGE.glob("*.json"):
            doc = self.load(file_path.stem)
            if doc:
                docs.append(doc)
        return docs