# Professional RAG and MCP Implementation

This document describes the professional implementations of RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol) utilities for the DoctorAI medical AI system.

## Overview

The utilities have been completely rewritten to provide enterprise-grade functionality with proper error handling, logging, configuration management, and professional coding standards.

## Table of Contents

- [RAG System (Retrieval-Augmented Generation)](#rag-system)
- [MCP System (Model Context Protocol)](#mcp-system)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Installation & Setup](#installation--setup)
- [API Reference](#api-reference)

## RAG System

### Features

The `MedicalRAGSystem` provides sophisticated document retrieval and context generation:

- **Professional Architecture**: Object-oriented design with proper error handling
- **Concurrent Processing**: Multi-threaded document loading for performance
- **Intelligent Chunking**: Smart text chunking with sentence boundary detection
- **Advanced Embeddings**: HuggingFace embeddings with normalization
- **FAISS Integration**: High-performance similarity search using FAISS
- **Context Management**: Intelligent context formatting for LLM consumption
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Batch Processing**: Efficient batch processing for large document sets

### Key Classes

#### `MedicalRAGSystem`
Main class for RAG operations with methods:
- `load_documents()`: Load PDF documents with concurrent processing
- `create_embeddings()`: Generate document embeddings in batches
- `build_index()`: Create FAISS index for similarity search
- `search_similar_documents()`: Find relevant documents for queries
- `get_context_for_query()`: Generate formatted context for LLM
- `initialize_system()`: One-command system initialization

#### `RAGError`
Custom exception class for RAG-specific errors.

### Usage Example

```python
from app.utils import MedicalRAGSystem, get_rag_system

# Initialize RAG system
rag = MedicalRAGSystem(
    pdf_folder="./library",
    chunk_size=1000,
    chunk_overlap=200
)

# Initialize (load docs + build index)
success = rag.initialize_system()

# Search for relevant documents
results = rag.search_similar_documents("diabetes symptoms", top_k=5)

# Get formatted context for LLM
context = rag.get_context_for_query("diabetes treatment", max_context_length=4000)
```

## MCP System

### Features

The `MedicalContextProtocol` provides sophisticated conversation management:

- **Message Management**: Structured message handling with roles and metadata
- **Context Window Management**: Intelligent token management with configurable limits
- **Multiple Strategies**: FIFO and Priority-based context management
- **Content Type Support**: Text, image, document, and medical data content types
- **Conversation Persistence**: Export/import conversations for persistence
- **Metadata Support**: Rich metadata support for conversation tracking
- **Token Estimation**: Smart token counting for context management
- **Content Handlers**: Pluggable handlers for different content types

### Key Classes

#### `MedicalContextProtocol`
Main class for context management with methods:
- `add_message()`: Add messages with automatic context management
- `get_conversation_history()`: Retrieve conversation in LLM-compatible format
- `clear_conversation()`: Clear context with optional system message preservation
- `export_conversation()`/`import_conversation()`: Persistence support
- `set_conversation_metadata()`: Manage conversation metadata

#### `Message`
Structured message representation with:
- Role (system, user, assistant, tool)
- Content and content type
- Timestamp and metadata
- Serialization support

#### `ContextWindow`
Token-aware message container with:
- Token counting and management
- Message addition/removal
- Overflow handling

#### Context Management Strategies
- **`FIFOContextManager`**: First-in-first-out message removal
- **`PriorityContextManager`**: Priority-based with system message preservation

### Usage Example

```python
from app.utils import MedicalContextProtocol, MessageRole, get_mcp

# Initialize MCP system
mcp = MedicalContextProtocol(
    max_context_tokens=8000,
    enable_persistence=True
)

# Add messages
mcp.add_message(MessageRole.USER, "I have chest pain")
mcp.add_message(MessageRole.ASSISTANT, "I understand your concern...")

# Get conversation history for LLM
history = mcp.get_conversation_history()

# Export for persistence
exported = mcp.export_conversation()
```

## Configuration

### Environment Variables

The professional implementation supports extensive configuration via environment variables:

#### RAG Configuration
```bash
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_TOP_K_RESULTS=5
RAG_SIMILARITY_THRESHOLD=0.7
RAG_MAX_CONTEXT_LENGTH=4000
RAG_BATCH_SIZE=32
RAG_MAX_WORKERS=4
```

#### MCP Configuration
```bash
MCP_MAX_CONTEXT_TOKENS=8000
MCP_TOKEN_BUFFER=1000
MCP_ENABLE_PERSISTENCE=True
MCP_CONTEXT_STRATEGY=priority
MCP_PRESERVE_SYSTEM_MESSAGES=True
```

#### Logging Configuration
```bash
LOG_LEVEL=INFO
LOG_FILE=logs/doctorai.log
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
```

### Configuration Methods

```python
from app.config import Config

# Get RAG configuration
rag_config = Config.get_rag_config()

# Get MCP configuration
mcp_config = Config.get_mcp_config()

# Initialize application
Config.init_app(app)
```

## Usage Examples

### Basic RAG Usage

```python
from app.utils import search_medical_documents, get_medical_context

# Simple search
results = search_medical_documents("hypertension treatment")

# Get formatted context
context = get_medical_context("diabetes management", max_length=3000)
```

### Basic MCP Usage

```python
from app.utils import add_user_message, add_assistant_message, get_chat_history

# Add messages
add_user_message("What are the symptoms of flu?")
add_assistant_message("Common flu symptoms include...")

# Get conversation
history = get_chat_history(last_n=10)
```

### Integrated Usage

```python
from app.utils import get_rag_system, get_mcp

# Get systems
rag = get_rag_system()
mcp = get_mcp()

# Process user query with RAG support
user_query = "Tell me about heart disease"
medical_context = rag.get_context_for_query(user_query)

# Add to conversation
mcp.add_message("user", user_query)
mcp.add_message("assistant", f"Based on medical literature: {medical_context}")
```

## Installation & Setup

### Dependencies

The professional implementation requires:

```bash
pip install numpy faiss-cpu langchain-community langchain-huggingface
```

### Directory Structure

Ensure these directories exist:
```
project/
├── library/          # PDF documents for RAG
├── logs/            # Log files
├── uploads/         # File uploads
└── instance/        # Database and instance files
```

### Initialization

```python
from app.config import Config

# Initialize logging and directories
Config.init_app(app)

# Initialize systems
from app.utils import get_rag_system, get_mcp
rag = get_rag_system()
mcp = get_mcp()
```

## API Reference

### RAG System API

#### `MedicalRAGSystem`

**Constructor**
```python
MedicalRAGSystem(
    model_name: str = None,
    pdf_folder: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
)
```

**Methods**
- `load_documents(max_workers: int = 4) -> List[Document]`
- `create_embeddings() -> np.ndarray`
- `build_index(embeddings: Optional[np.ndarray] = None) -> faiss.Index`
- `search_similar_documents(query: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Tuple[str, float]]`
- `get_context_for_query(query: str, max_context_length: int = 4000, top_k: int = 5) -> str`
- `initialize_system() -> bool`

#### Global Functions
- `get_rag_system() -> MedicalRAGSystem`
- `search_medical_documents(query: str, top_k: int = 5) -> List[Tuple[str, float]]`
- `get_medical_context(query: str, max_length: int = 4000) -> str`

### MCP System API

#### `MedicalContextProtocol`

**Constructor**
```python
MedicalContextProtocol(
    max_context_tokens: int = 4000,
    context_manager: Optional[ContextManager] = None,
    enable_persistence: bool = True
)
```

**Methods**
- `add_message(role: Union[MessageRole, str], content: str, content_type: Union[ContentType, str] = ContentType.TEXT, metadata: Optional[Dict[str, Any]] = None) -> bool`
- `get_conversation_history(include_system: bool = False, last_n_messages: Optional[int] = None) -> List[Dict[str, Any]]`
- `get_context_summary() -> Dict[str, Any]`
- `clear_conversation(preserve_system: bool = True)`
- `export_conversation() -> Dict[str, Any]`
- `import_conversation(conversation_data: Dict[str, Any])`
- `register_content_handler(content_type: ContentType, handler: Callable[[Message], None])`
- `set_conversation_metadata(key: str, value: Any)`
- `get_conversation_metadata(key: str, default: Any = None) -> Any`

#### Global Functions
- `get_mcp() -> MedicalContextProtocol`
- `add_user_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> bool`
- `add_assistant_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> bool`
- `get_chat_history(last_n: Optional[int] = None) -> List[Dict[str, Any]]`

## Error Handling

### Custom Exceptions
- **`RAGError`**: Raised for RAG-specific errors
- **`MCPError`**: Raised for MCP-specific errors

### Logging
Both systems provide comprehensive logging:
- **INFO**: Normal operations and status
- **DEBUG**: Detailed debugging information
- **WARNING**: Non-critical issues
- **ERROR**: Error conditions with context

### Example Error Handling

```python
from app.utils import RAGError, MCPError

try:
    rag = MedicalRAGSystem()
    rag.initialize_system()
except RAGError as e:
    logger.error(f"RAG initialization failed: {e}")
    
try:
    mcp = MedicalContextProtocol()
    mcp.add_message("user", "Hello")
except MCPError as e:
    logger.error(f"MCP operation failed: {e}")
```

## Performance Considerations

### RAG System
- Use appropriate batch sizes for embedding generation
- Consider GPU acceleration for large document sets
- Monitor memory usage with large PDF collections
- Use concurrent processing for document loading

### MCP System
- Monitor token usage to prevent context overflow
- Choose appropriate context management strategy
- Use metadata efficiently
- Consider conversation persistence for long sessions

## Security Considerations

### Medical Data
- All systems include medical disclaimers
- Professional consultation reminders are built-in
- Safety filters can be enabled via configuration
- Conversation metadata supports patient tracking

### Data Privacy
- No data is stored externally without explicit configuration
- Conversation persistence is optional
- Logging can be configured to exclude sensitive data
- Export/import supports data governance requirements

## Migration from Legacy Code

The new implementation is backward-compatible but offers enhanced functionality:

### Old RAG Usage
```python
# Legacy
from app.utils.rag import search_docs
results = search_docs(query, embedding_model, index, texts)
```

### New RAG Usage
```python
# Professional
from app.utils import search_medical_documents
results = search_medical_documents(query, top_k=5)
```

### Benefits of Migration
1. **Error Handling**: Comprehensive error handling and logging
2. **Performance**: Optimized processing and concurrent operations
3. **Flexibility**: Configurable parameters and strategies
4. **Maintainability**: Object-oriented design and documentation
5. **Features**: Advanced context management and persistence
6. **Standards**: Professional coding standards and best practices

## Support and Maintenance

### Logging
Monitor the `logs/doctorai.log` file for system status and errors.

### Configuration
Use environment variables for production deployment configuration.

### Updates
The modular design allows for easy updates and feature additions.

### Troubleshooting
1. Check log files for detailed error information
2. Verify PDF documents are accessible in the library folder
3. Ensure proper environment variable configuration
4. Monitor memory usage for large document sets

---

© 2025 DoctorAI. All rights reserved.
This implementation provides professional-grade RAG and MCP systems for medical AI applications. 