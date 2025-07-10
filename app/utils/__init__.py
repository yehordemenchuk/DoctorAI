"""
© 2025 DoctorAI. All rights reserved. 
Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation.

Professional utilities package for DoctorAI medical AI system.
"""

# RAG System exports
from .rag import (
    MedicalRAGSystem,
    RAGError,
    get_rag_system,
    search_medical_documents,
    get_medical_context
)

# MCP (Model Context Protocol) exports
from .mcp import (
    MedicalContextProtocol,
    Message,
    MessageRole,
    ContentType,
    ContextWindow,
    MCPError,
    get_mcp,
    add_user_message,
    add_assistant_message,
    get_chat_history
)

# Other utilities
from .doctor import get_response, personal_consultation, analyzing_medical_document
from .file_reader import *
from .ocr import *

__all__ = [
    # RAG System
    'MedicalRAGSystem',
    'RAGError',
    'get_rag_system',
    'search_medical_documents',
    'get_medical_context',
    
    # MCP System
    'MedicalContextProtocol',
    'Message',
    'MessageRole',
    'ContentType',
    'ContextWindow',
    'MCPError',
    'get_mcp',
    'add_user_message',
    'add_assistant_message',
    'get_chat_history',
    
    # Doctor utilities
    'get_response',
    'personal_consultation',
    'analyzing_medical_document',
]
