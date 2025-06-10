import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).parent.parent.parent
DOWNLOAD_DIR = Path("D:/VKR/download")
DATA_DIR = DOWNLOAD_DIR / 'data'
HTML_DIR = DATA_DIR / 'html'
PDF_DIR = DATA_DIR / 'pdf'
LOG_DIR = DOWNLOAD_DIR / 'logs'
CACHE_DIR = DOWNLOAD_DIR / 'cache'
MODEL_DIR = DOWNLOAD_DIR / 'models'
MODELS_DIR = Path("D:/VKR/download/models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = str(MODELS_DIR)
os.environ['HF_HOME'] = str(MODELS_DIR)

load_dotenv()

LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

VECTOR_DB_CONFIG = {
    "persist_directory": str(BASE_DIR / "data" / "chroma"),
    "collection_name": "documents",
    "distance_metric": "cosine",
    "max_results": 5
}

EMBEDDING_CONFIG = {
    'model_name': 'sberbank-ai/sbert_large_nlu_ru',
    'device': 'cpu',
    'cache_folder': str(MODELS_DIR)
}

CACHE_CONFIG = {
    "max_size": 256 * 1024 * 1024,
    "max_cache_size": 256 * 1024 * 1024,
    "ttl": 3600,
    "cleanup_interval": 300,
    "transformers_cache": str(MODEL_DIR),
    "hf_home": str(CACHE_DIR),
    "chroma_cache": str(CACHE_DIR / "chroma")
}

FILE_CONFIG = {
    'supported_extensions': {'.html', '.pdf'},
    'max_file_size': 10 * 1024 * 1024
}

class Config:
   
    def __init__(self):

        self.BASE_DIR = BASE_DIR
        self.MODELS_DIR = MODELS_DIR
        self.DOCUMENTS_DIR = PDF_DIR
        self.DATA_DIR = DATA_DIR
        
        self.DB_PATH = self.DATA_DIR / "chroma"
        
        self.EMBEDDING_MODEL = EMBEDDING_CONFIG['model_name']
        self.EMBEDDING_CONFIG = EMBEDDING_CONFIG
        
 
        self.SEARCH_CONFIG = {
            'k': 5, 
            'score_threshold': 0.2  #порог релевантности
        }
        
        self.ensure_directories()
        
    def ensure_directories(self):
        for directory in [self.MODELS_DIR, self.DOCUMENTS_DIR, self.DATA_DIR, self.DB_PATH]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_model_path(self, model_name: str) -> Path:
        return self.MODELS_DIR / model_name
        
    def get_document_path(self, filename: str) -> Path:
        return self.DOCUMENTS_DIR / filename

    DOWNLOAD_DIR = DOWNLOAD_DIR
    HTML_DIR = HTML_DIR
    PDF_DIR = PDF_DIR
    LOG_DIR = LOG_DIR
    CACHE_DIR = CACHE_DIR
    MODEL_DIR = MODEL_DIR
    LOG_CONFIG = LOG_CONFIG
    VECTOR_DB_CONFIG = VECTOR_DB_CONFIG
    FILE_CONFIG = FILE_CONFIG
    CACHE_CONFIG = CACHE_CONFIG

    #Настройки базы данных
    DATABASE_CONFIG = {
        'persist_directory': str(BASE_DIR / "data" / "chroma"),
        'collection_name': "documents"
    }

    #Настройки Telegram-бота
    TELEGRAM_BOT_TOKEN = "7007005918:AAFZlCU1gK-in9mBcsRsnbjmX85lIecizPg"
    TELEGRAM_BOT_CONFIG = {
        'token': TELEGRAM_BOT_TOKEN,
        'webhook_url': os.environ.get("TELEGRAM_WEBHOOK_URL", ""),
        'webhook_port': int(os.environ.get("TELEGRAM_WEBHOOK_PORT", "8443")),
        'allowed_users': os.environ.get("TELEGRAM_ALLOWED_USERS", "").split(",") if os.environ.get("TELEGRAM_ALLOWED_USERS") else []
    }

    def get_parser_config(self, parser_type: str) -> Dict[str, Any]:
        configs = {
            "html": {
                "extract_metadata": True,
                "extract_text": True,
                "clean_text": True
            },
            "pdf": {
                "extract_metadata": True,
                "extract_text": True,
                "clean_text": True
            }
        }
        return configs.get(parser_type, {})

config = Config()

def ensure_directories():
    directories = [
        config.MODELS_DIR,
        config.DOCUMENTS_DIR,
        Path(config.DATABASE_CONFIG['persist_directory']).parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

ensure_directories()