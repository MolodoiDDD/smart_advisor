import logging
import sys
from pythonjsonlogger import jsonlogger
from smart_advisor.config import LOG_LEVEL, LOG_FORMAT, LOG_DIR
import os

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    if LOG_FORMAT == 'json':
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, 'app.log'),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)