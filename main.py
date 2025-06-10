import asyncio
import logging
from smart_advisor.core.file_loader import FileLoader
from smart_advisor.core.database import VectorDatabase
from smart_advisor.services.query_processor import QueryProcessor
from smart_advisor.core.embeddings import EmbeddingService
from smart_advisor.telegram_bot import TelegramBot


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        embedding_service = EmbeddingService()
        file_loader = FileLoader("F:/VKR/download/data/pdf")
        vector_db = VectorDatabase(embedding_service)

        documents = file_loader.load_documents()
        logger.info(f"Successfully processed {len(documents)} documents")

        vector_db.add_documents(documents)

        query_processor = QueryProcessor(vector_db)

        bot = TelegramBot(query_processor=query_processor)
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        if 'vector_db' in locals():
            vector_db.clear()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")