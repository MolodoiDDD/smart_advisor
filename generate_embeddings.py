import logging
import sys
import os
from tqdm import tqdm
from smart_advisor.services.file_loader import FileLoader
from smart_advisor.core.embeddings import EmbeddingService
from smart_advisor.core.database import VectorDatabase
from smart_advisor.core.models import Document
from smart_advisor.config import settings
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('embeddings_generation.log')
    ]
)
logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    text = ' '.join(text.split())
    text = text.lower()
    return text.strip()

def split_into_paragraphs(text: str) -> list:
    paragraphs = [p.strip() for p in text.split('\n\n')]
    return [p for p in paragraphs if p]

def process_document(doc: Document, embedder: EmbeddingService) -> list:
#Обработка одного документа: разбиение наабзацы и генерация эмбеддингов ((( ПОСМОТРЕТЬ ЕЩЕ РАЗ
    processed_docs = []
    paragraphs = split_into_paragraphs(doc.text)
    
    for paragraph in paragraphs:
        if paragraph.strip():
            new_doc = Document(
                text=preprocess_text(paragraph),
                metadata=doc.metadata.copy(),
                id=str(uuid.uuid4())
            )
            embedding = embedder.generate_single(new_doc.text)
            new_doc.set_embedding(embedding)
            processed_docs.append(new_doc)
    
    return processed_docs

def generate_embeddings():
    try:
        print("\n=== Генерация эмбеддингов ===")
        logger.info("Начало генерации эмбеддингов...")

        for dir_path in [settings.DATA_DIR, settings.HTML_DIR, settings.PDF_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"Создана директория: {dir_path}")
                logger.info(f"Создана директория: {dir_path}")

        print("\nСоздание сервисов...")
        file_loader = FileLoader()
        print("✓ FileLoader создан")
        
        embedder = EmbeddingService()
        print("✓ EmbeddingService создан")
        
        vector_db = VectorDatabase(embedder)
        print("✓ VectorDatabase создана")

        print("\nЗагрузка документов...")
        documents = file_loader.load_documents()
        
        if not documents:
            print("⚠️ Документы не найдены")
            logger.warning("Документы не найдены")
            return
        
        print(f"✓ Загружено {len(documents)} документов")
        logger.info(f"Загружено {len(documents)} документов")

        print("\nГенерация эмбеддингов и добавление в базу данных...")
        all_processed_docs = []
        
        for i, doc in enumerate(tqdm(documents, desc="Обработка документов")):
            try:
                print(f"\nОбработка документа {i+1}/{len(documents)}: {doc.metadata.get('source', 'unknown')}")
                processed_docs = process_document(doc, embedder)
                all_processed_docs.extend(processed_docs)
                print(f"✓ Обработано {len(processed_docs)} абзацев")
            except Exception as e:
                print(f"⚠️ Ошибка при обработке документа {i+1}: {str(e)}")
                logger.error(f"Ошибка при обработке документа {i+1}: {str(e)}")
                continue
        
        if not all_processed_docs:
            print("⚠️ Не удалось обработать ни один документ")
            logger.error("Не удалось обработать ни один документ")
            return

        print("\nОбновление базы данных...")
        try:
            vector_db.clear()
            vector_db.add_documents(all_processed_docs)
            print(f"✓ Добавлено {len(all_processed_docs)} документов в базу данных")
            logger.info(f"Добавлено {len(all_processed_docs)} документов в базу данных")
        except Exception as e:
            print(f"❌ Ошибка при обновлении базы данных: {str(e)}")
            logger.error(f"Ошибка при обновлении базы данных: {str(e)}")
            return
        
        print("\n=== Генерация эмбеддингов успешно завершена ===")
        logger.info("Генерация эмбеддингов успешно завершена")
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        logger.error(f"Критическая ошибка: {str(e)}")
        return

if __name__ == "__main__":
    generate_embeddings() 