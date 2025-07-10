"""
© 2025 DoctorAI. All rights reserved. 
Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation.
"""

import os
import secrets
import logging
import logging.handlers
from pathlib import Path


class Config:
    """Professional configuration class for DoctorAI medical AI system."""
    
    SECRET_KEY = secrets.token_hex(16)
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    PDF_FOLDER = os.environ.get('PDF_FOLDER', 'library')
    STATIC_FOLDER = os.environ.get('STATIC_FOLDER', 'static')
    TEMPLATES_FOLDER = os.environ.get('TEMPLATES_FOLDER', 'templates')
    
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', 
        'sqlite:///instance/database.db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    MODEL = os.environ.get('AI_MODEL', 'gpt-4o')
    FALLBACK_MODEL = os.environ.get('FALLBACK_MODEL', 'gpt-3.5-turbo')
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
    REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', '30'))
    
    TRANSFORMER_EMBEDDINGS_MODEL_NAME = os.environ.get(
        'EMBEDDINGS_MODEL', 
        'sentence-transformers/all-MiniLM-L6-v2'
    )
    RAG_CHUNK_SIZE = int(os.environ.get('RAG_CHUNK_SIZE', '1000'))
    RAG_CHUNK_OVERLAP = int(os.environ.get('RAG_CHUNK_OVERLAP', '200'))
    RAG_TOP_K_RESULTS = int(os.environ.get('RAG_TOP_K_RESULTS', '5'))
    RAG_SIMILARITY_THRESHOLD = float(os.environ.get('RAG_SIMILARITY_THRESHOLD', '0.7'))
    RAG_MAX_CONTEXT_LENGTH = int(os.environ.get('RAG_MAX_CONTEXT_LENGTH', '4000'))
    RAG_BATCH_SIZE = int(os.environ.get('RAG_BATCH_SIZE', '32'))
    RAG_MAX_WORKERS = int(os.environ.get('RAG_MAX_WORKERS', '4'))
    
    MCP_MAX_CONTEXT_TOKENS = int(os.environ.get('MCP_MAX_CONTEXT_TOKENS', '8000'))
    MCP_TOKEN_BUFFER = int(os.environ.get('MCP_TOKEN_BUFFER', '1000'))
    MCP_ENABLE_PERSISTENCE = os.environ.get('MCP_ENABLE_PERSISTENCE', 'True').lower() == 'true'
    MCP_CONTEXT_STRATEGY = os.environ.get('MCP_CONTEXT_STRATEGY', 'priority')
    MCP_PRESERVE_SYSTEM_MESSAGES = os.environ.get('MCP_PRESERVE_SYSTEM_MESSAGES', 'True').lower() == 'true'
    
    TESSERACT_CMD = os.environ.get(
        'TESSERACT_CMD', 
        r'D:\Tesseract-OCR\tesseract.exe'
    )
    OCR_LANGUAGES = os.environ.get('OCR_LANGUAGES', 'eng+rus+fra+spa+deu')
    OCR_PAGE_SEGMENTATION = int(os.environ.get('OCR_PAGE_SEGMENTATION', '6'))
    OCR_ENGINE_MODE = int(os.environ.get('OCR_ENGINE_MODE', '3'))
    
    MEDICAL_DISCLAIMER_REQUIRED = True
    ENABLE_SAFETY_FILTERS = os.environ.get('ENABLE_SAFETY_FILTERS', 'True').lower() == 'true'
    MAX_CONSULTATION_LENGTH = int(os.environ.get('MAX_CONSULTATION_LENGTH', '5000'))
    REQUIRE_PROFESSIONAL_DISCLAIMER = True
    
    MAX_FILE_SIZE_MB = int(os.environ.get('MAX_FILE_SIZE_MB', '50'))
    MAX_CONCURRENT_REQUESTS = int(os.environ.get('MAX_CONCURRENT_REQUESTS', '10'))
    CACHE_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT', '3600'))
    
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    LOG_FORMAT = os.environ.get(
        'LOG_FORMAT',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/doctorai.log')
    LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', '10485760'))
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', '5'))
    
    SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', '3600'))
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', '60'))
    ENABLE_CSRF_PROTECTION = os.environ.get('ENABLE_CSRF_PROTECTION', 'True').lower() == 'true'
    
    API_VERSION = os.environ.get('API_VERSION', 'v1')
    API_RATE_LIMIT = os.environ.get('API_RATE_LIMIT', '100/hour')
    
    @classmethod
    def init_app(cls, app):
        """Initialize application with configuration."""
        dirs_to_create = [
            cls.UPLOAD_FOLDER,
            cls.PDF_FOLDER,
            'logs',
            'instance'
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
        
        cls.setup_logging()
    
    @classmethod
    def setup_logging(cls):
        """Setup professional logging configuration."""
        log_dir = os.path.dirname(cls.LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(
                    cls.LOG_FILE,
                    maxBytes=cls.LOG_MAX_BYTES,
                    backupCount=cls.LOG_BACKUP_COUNT
                )
            ]
        )
        
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        app_logger = logging.getLogger('doctorai')
        app_logger.setLevel(getattr(logging, cls.LOG_LEVEL))
    
    @classmethod
    def get_rag_config(cls) -> dict:
        """Get RAG system configuration."""
        return {
            'model_name': cls.TRANSFORMER_EMBEDDINGS_MODEL_NAME,
            'pdf_folder': cls.PDF_FOLDER,
            'chunk_size': cls.RAG_CHUNK_SIZE,
            'chunk_overlap': cls.RAG_CHUNK_OVERLAP,
            'top_k': cls.RAG_TOP_K_RESULTS,
            'similarity_threshold': cls.RAG_SIMILARITY_THRESHOLD,
            'max_context_length': cls.RAG_MAX_CONTEXT_LENGTH,
            'batch_size': cls.RAG_BATCH_SIZE,
            'max_workers': cls.RAG_MAX_WORKERS
        }
    
    @classmethod
    def get_mcp_config(cls) -> dict:
        """Get MCP system configuration."""
        return {
            'max_context_tokens': cls.MCP_MAX_CONTEXT_TOKENS,
            'token_buffer': cls.MCP_TOKEN_BUFFER,
            'enable_persistence': cls.MCP_ENABLE_PERSISTENCE,
            'context_strategy': cls.MCP_CONTEXT_STRATEGY,
            'preserve_system_messages': cls.MCP_PRESERVE_SYSTEM_MESSAGES
        }


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', Config.SQLALCHEMY_DATABASE_URI)


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}