import os
import logging
from typing import List, Optional
from pathlib import Path
from PyPDF2 import PdfReader

from ..models import Document
from ..config.config import config

logger = logging.getLogger(__name__)

class FileLoader:
    
    def __init__(self, documents_dir: Optional[str] = None):
        self.documents_dir = Path(documents_dir) if documents_dir else config.DOCUMENTS_DIR
        logger.info(f"Инициализация FileLoader с директорией: {self.documents_dir}")
        
    def load_documents(self) -> List[Document]:
        documents = []
        
        if not self.documents_dir.exists():
            logger.warning(f"Директория {self.documents_dir} не существует")
            return documents
            
        logger.info(f"Сканирование директории: {self.documents_dir}")
        
        for file_path in self.documents_dir.glob("**/*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.pdf':
                        logger.info(f"Обработка PDF файла: {file_path}")
                        doc = self.load_pdf(file_path)
                    elif file_path.suffix.lower() == '.txt':
                        logger.info(f"Обработка текстового файла: {file_path}")
                        doc = self.load_txt(file_path)
                    else:
                        continue
                        
                    if doc:
                        documents.append(doc)
                        logger.info(f"Успешно загружен файл: {file_path}")
                    else:
                        logger.warning(f"Не удалось загрузить файл: {file_path}")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке файла {file_path}: {e}")
                    
        logger.info(f"Всего загружено документов: {len(documents)}")
        return documents
        
    def load_pdf(self, file_path: Path) -> Optional[Document]:
        try:
            logger.info(f"Загрузка PDF файла: {file_path}")
            reader = PdfReader(str(file_path))
            logger.info(f"PDF файл загружен, количество страниц: {len(reader.pages)}")
            
            text = ""
            for i, page in enumerate(reader.pages, 1):
                logger.info(f"Извлечение текста со страницы {i}/{len(reader.pages)}")
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logger.warning(f"Текст не извлечен со страницы {i}")
            
            if not text.strip():
                logger.warning(f"Текст не извлечен из PDF файла: {file_path}")
                return None
            
            metadata = reader.metadata
            title = metadata.get('/Title', file_path.name)
            
            logger.info(f"Успешно обработан PDF файл: {file_path}")
            logger.info(f"Длина извлеченного текста: {len(text)} символов")
            
            return Document(
                id=str(file_path),
                text=text,
                metadata={
                    'source': str(file_path),
                    'title': title,
                    'type': 'pdf',
                    'pages': len(reader.pages)
                }
            )
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке PDF файла {file_path}: {e}", exc_info=True)
            return None
            
    def load_txt(self, file_path: Path) -> Optional[Document]:
        try:
            logger.info(f"Загрузка текстового файла: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            if not text.strip():
                logger.warning(f"Текст не извлечен из файла: {file_path}")
                return None
                
            logger.info(f"Успешно обработан текстовый файл: {file_path}")
            logger.info(f"Длина извлеченного текста: {len(text)} символов")
            
            return Document(
                id=str(file_path),
                text=text,
                metadata={
                    'source': str(file_path),
                    'title': file_path.name,
                    'type': 'txt'
                }
            )
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке текстового файла {file_path}: {e}", exc_info=True)
            return None 