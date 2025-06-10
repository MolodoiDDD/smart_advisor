import os
import logging
from typing import List, Optional
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from ..models import Document
from ..config import settings

logger = logging.getLogger(__name__)

class FileLoader:
  
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.html_dir = settings.HTML_DIR
        self.pdf_dir = settings.PDF_DIR
        self.supported_extensions = {'.html', '.pdf'}
        
    def load_documents(self) -> List[Document]:
        documents = []
        
        if os.path.exists(self.html_dir):
            logger.info(f"Scanning HTML directory: {self.html_dir}")
            html_files = [f for f in os.listdir(self.html_dir) if f.endswith('.html')]
            logger.info(f"Found {len(html_files)} HTML files")
            
            for filename in html_files:
                file_path = os.path.join(self.html_dir, filename)
                logger.info(f"Processing HTML file: {filename}")
                doc = self.load_html(file_path)
                if doc:
                    documents.append(doc)
                    logger.info(f"Successfully loaded HTML file: {filename}")
                else:
                    logger.warning(f"Failed to load HTML file: {filename}")
        else:
            logger.warning(f"HTML directory does not exist: {self.html_dir}")
        
        if os.path.exists(self.pdf_dir):
            logger.info(f"Scanning PDF directory: {self.pdf_dir}")
            pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            for filename in pdf_files:
                file_path = os.path.join(self.pdf_dir, filename)
                logger.info(f"Processing PDF file: {filename}")
                doc = self.load_pdf(file_path)
                if doc:
                    documents.append(doc)
                    logger.info(f"Successfully loaded PDF file: {filename}")
                else:
                    logger.warning(f"Failed to load PDF file: {filename}")
        else:
            logger.warning(f"PDF directory does not exist: {self.pdf_dir}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def load_html(self, file_path: str) -> Optional[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            title = soup.title.string if soup.title else os.path.basename(file_path)
            
            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
            
            return Document(
                id=os.path.basename(file_path),
                text=text,
                metadata={
                    'source': file_path,
                    'title': title,
                    'type': 'html'
                }
            )
            
        except Exception as e:
            logger.error(f"Error loading HTML file {file_path}: {str(e)}")
            return None
            
    def load_pdf(self, file_path: str) -> Optional[Document]:
        try:
            logger.info(f"Loading PDF file: {file_path}")
            reader = PdfReader(file_path)
            logger.info(f"PDF file loaded, number of pages: {len(reader.pages)}")
            
            text = ""
            for i, page in enumerate(reader.pages, 1):
                logger.info(f"Extracting text from page {i}/{len(reader.pages)}")
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logger.warning(f"No text extracted from page {i}")
            
            if not text.strip():
                logger.warning(f"No text extracted from PDF file: {file_path}")
                return None
            
            metadata = reader.metadata
            title = metadata.get('/Title', os.path.basename(file_path))
            
            logger.info(f"Successfully processed PDF file: {file_path}")
            logger.info(f"Extracted text length: {len(text)} characters")
            
            return Document(
                id=os.path.basename(file_path),
                text=text,
                metadata={
                    'source': file_path,
                    'title': title,
                    'type': 'pdf',
                    'pages': len(reader.pages)
                }
            )
            
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {str(e)}", exc_info=True)
            return None

    def generate_embeddings(self, documents: List[Document]) -> List[Document]:
        from ..core.embeddings import EmbeddingService
        embedding_service = EmbeddingService()
        for doc in documents:
            embedding = embedding_service.generate_single(doc.text)
            doc.set_embedding(embedding)
        return documents