

import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from app.config import Config

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Enumeration for message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(Enum):
    """Enumeration for content types."""
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    MEDICAL_DATA = "medical_data"


class MCPError(Exception):
    """Custom exception for MCP-related errors."""
    pass


@dataclass
class Message:
    """
    Represents a single message in the conversation context.
    """
    role: MessageRole
    content: str
    content_type: ContentType = ContentType.TEXT
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "role": self.role.value,
            "content": self.content,
            "content_type": self.content_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            content_type=ContentType(data.get("content_type", "text")),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            metadata=data.get("metadata", {})
        )


@dataclass
class ContextWindow:
    """
    Represents a context window with token management.
    """
    max_tokens: int
    current_tokens: int = 0
    messages: List[Message] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation: 1 token ≈ 4 characters).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def can_add_message(self, message: Message) -> bool:
        """
        Check if a message can be added without exceeding token limit.
        
        Args:
            message: Message to check
            
        Returns:
            True if message can be added, False otherwise
        """
        estimated_tokens = self.estimate_tokens(message.content)
        return self.current_tokens + estimated_tokens <= self.max_tokens
    
    def add_message(self, message: Message) -> bool:
        """
        Add a message to the context window.
        
        Args:
            message: Message to add
            
        Returns:
            True if message was added, False if it would exceed token limit
        """
        if not self.can_add_message(message):
            return False
        
        self.messages.append(message)
        self.current_tokens += self.estimate_tokens(message.content)
        return True
    
    def remove_oldest_messages(self, count: int = 1) -> List[Message]:
        """
        Remove the oldest messages from the context.
        
        Args:
            count: Number of messages to remove
            
        Returns:
            List of removed messages
        """
        removed = []
        for _ in range(min(count, len(self.messages))):
            if self.messages:
                message = self.messages.pop(0)
                self.current_tokens -= self.estimate_tokens(message.content)
                removed.append(message)
        return removed
    
    def clear(self):
        """Clear all messages from the context window."""
        self.messages.clear()
        self.current_tokens = 0


class ContextManager(ABC):
    """Abstract base class for context management strategies."""
    
    @abstractmethod
    def manage_context(self, context_window: ContextWindow, new_message: Message) -> bool:
        """
        Manage context when adding a new message.
        
        Args:
            context_window: Current context window
            new_message: New message to add
            
        Returns:
            True if message was successfully added
        """
        pass


class FIFOContextManager(ContextManager):
    """First-In-First-Out context management strategy."""
    
    def manage_context(self, context_window: ContextWindow, new_message: Message) -> bool:
        """
        Add message using FIFO strategy - remove oldest messages if needed.
        
        Args:
            context_window: Current context window
            new_message: New message to add
            
        Returns:
            True if message was successfully added
        """
        if context_window.can_add_message(new_message):
            return context_window.add_message(new_message)
        
        while context_window.messages and not context_window.can_add_message(new_message):
            removed = context_window.remove_oldest_messages(1)
            logger.debug(f"Removed message to make space: {removed[0].content[:50]}...")
        
        return context_window.add_message(new_message)


class PriorityContextManager(ContextManager):
    """Priority-based context management that preserves important messages."""
    
    def __init__(self, preserve_system_messages: bool = True):
        self.preserve_system_messages = preserve_system_messages
    
    def manage_context(self, context_window: ContextWindow, new_message: Message) -> bool:
        """
        Add message using priority strategy - preserve important messages.
        
        Args:
            context_window: Current context window
            new_message: New message to add
            
        Returns:
            True if message was successfully added
        """
        if context_window.can_add_message(new_message):
            return context_window.add_message(new_message)
        
        messages_to_remove = []
        for i, message in enumerate(context_window.messages):
            if self.preserve_system_messages and message.role == MessageRole.SYSTEM:
                continue
            messages_to_remove.append(i)
        
        for i in reversed(messages_to_remove):
            if context_window.can_add_message(new_message):
                break
            removed_message = context_window.messages.pop(i)
            context_window.current_tokens -= context_window.estimate_tokens(removed_message.content)
            logger.debug(f"Removed non-priority message: {removed_message.content[:50]}...")
        
        return context_window.add_message(new_message)


class MedicalContextProtocol:
    """
    Professional Model Context Protocol implementation for medical AI systems.
    
    This class manages conversation context, message history, and provides
    intelligent context management for medical AI interactions.
    """
    
    def __init__(
        self,
        max_context_tokens: int = 4000,
        context_manager: Optional[ContextManager] = None,
        enable_persistence: bool = True
    ):
        """
        Initialize the Medical Context Protocol.
        
        Args:
            max_context_tokens: Maximum tokens in context window
            context_manager: Context management strategy
            enable_persistence: Whether to enable conversation persistence
        """
        self.context_window = ContextWindow(max_tokens=max_context_tokens)
        self.context_manager = context_manager or FIFOContextManager()
        self.enable_persistence = enable_persistence
        
        self.conversation_id: Optional[str] = None
        self.session_metadata: Dict[str, Any] = {}
        self.message_handlers: Dict[ContentType, Callable] = {}
        
        self._initialize_system_context()
        
        logger.info(f"Initialized MedicalContextProtocol with {max_context_tokens} token limit")
    
    def _initialize_system_context(self):
        """Initialize system context with medical AI guidelines."""
        system_prompt = """You are a professional medical AI assistant. Follow these guidelines:
        
        1. Provide accurate, evidence-based medical information
        2. Always recommend consulting healthcare professionals for diagnosis and treatment
        3. Be empathetic and professional in your responses
        4. Clearly state limitations and when to seek emergency care
        5. Respect patient privacy and confidentiality
        6. Use clear, understandable language while maintaining medical accuracy
        
        Remember: You are not a replacement for professional medical consultation."""
        
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=system_prompt,
            metadata={"priority": "high", "permanent": True}
        )
        
        self.context_window.add_message(system_message)
    
    def add_message(
        self,
        role: Union[MessageRole, str],
        content: str,
        content_type: Union[ContentType, str] = ContentType.TEXT,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to the conversation context.
        
        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            content_type: Type of content
            metadata: Additional metadata
            
        Returns:
            True if message was added successfully
            
        Raises:
            MCPError: If message cannot be processed
        """
        try:
            if isinstance(role, str):
                role = MessageRole(role.lower())
            if isinstance(content_type, str):
                content_type = ContentType(content_type.lower())
            
            message = Message(
                role=role,
                content=content,
                content_type=content_type,
                metadata=metadata or {}
            )
            
            success = self.context_manager.manage_context(self.context_window, message)
            
            if success:
                logger.debug(f"Added {role.value} message with {len(content)} characters")
                
                if content_type in self.message_handlers:
                    self.message_handlers[content_type](message)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            raise MCPError(f"Failed to add message: {str(e)}")
    
    def get_conversation_history(
        self,
        include_system: bool = False,
        last_n_messages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history in a format suitable for LLM APIs.
        
        Args:
            include_system: Whether to include system messages
            last_n_messages: Limit to last N messages
            
        Returns:
            List of message dictionaries
        """
        messages = self.context_window.messages
        
        if not include_system:
            messages = [msg for msg in messages if msg.role != MessageRole.SYSTEM]
        
        if last_n_messages:
            messages = messages[-last_n_messages:]
        
        return [
            {
                "role": msg.role.value,
                "content": msg.content
            }
            for msg in messages
        ]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Returns:
            Dictionary with context information
        """
        return {
            "total_messages": len(self.context_window.messages),
            "current_tokens": self.context_window.current_tokens,
            "max_tokens": self.context_window.max_tokens,
            "token_utilization": self.context_window.current_tokens / self.context_window.max_tokens,
            "conversation_id": self.conversation_id,
            "session_metadata": self.session_metadata
        }
    
    def clear_conversation(self, preserve_system: bool = True):
        """
        Clear the conversation context.
        
        Args:
            preserve_system: Whether to preserve system messages
        """
        if preserve_system:
            system_messages = [
                msg for msg in self.context_window.messages 
                if msg.role == MessageRole.SYSTEM
            ]
            self.context_window.clear()
            for msg in system_messages:
                self.context_window.add_message(msg)
        else:
            self.context_window.clear()
        
        logger.info("Cleared conversation context")
    
    def export_conversation(self) -> Dict[str, Any]:
        """
        Export the conversation for persistence or analysis.
        
        Returns:
            Dictionary containing full conversation data
        """
        return {
            "conversation_id": self.conversation_id,
            "session_metadata": self.session_metadata,
            "messages": [msg.to_dict() for msg in self.context_window.messages],
            "context_summary": self.get_context_summary(),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_conversation(self, conversation_data: Dict[str, Any]):
        """
        Import a previously exported conversation.
        
        Args:
            conversation_data: Exported conversation data
            
        Raises:
            MCPError: If import fails
        """
        try:
            self.context_window.clear()
            self.conversation_id = conversation_data.get("conversation_id")
            self.session_metadata = conversation_data.get("session_metadata", {})
            
            for msg_data in conversation_data.get("messages", []):
                message = Message.from_dict(msg_data)
                self.context_window.add_message(message)
            
            logger.info(f"Imported conversation with {len(self.context_window.messages)} messages")
            
        except Exception as e:
            logger.error(f"Failed to import conversation: {str(e)}")
            raise MCPError(f"Failed to import conversation: {str(e)}")
    
    def register_content_handler(
        self,
        content_type: ContentType,
        handler: Callable[[Message], None]
    ):
        """
        Register a handler for specific content types.
        
        Args:
            content_type: Content type to handle
            handler: Handler function that takes a Message
        """
        self.message_handlers[content_type] = handler
        logger.debug(f"Registered handler for {content_type.value} content")
    
    def set_conversation_metadata(self, key: str, value: Any):
        """
        Set metadata for the current conversation session.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.session_metadata[key] = value
    
    def get_conversation_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata for the current conversation session.
        
        Args:
            key: Metadata key
            default: Default value if key doesn't exist
            
        Returns:
            Metadata value or default
        """
        return self.session_metadata.get(key, default)


_mcp_instance: Optional[MedicalContextProtocol] = None


def get_mcp() -> MedicalContextProtocol:
    """
    Get or create the global MCP instance.
    
    Returns:
        MedicalContextProtocol instance
    """
    global _mcp_instance
    
    if _mcp_instance is None:
        _mcp_instance = MedicalContextProtocol()
    
    return _mcp_instance


def add_user_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to add a user message.
    
    Args:
        content: Message content
        metadata: Optional metadata
        
    Returns:
        True if message was added successfully
    """
    mcp = get_mcp()
    return mcp.add_message(MessageRole.USER, content, metadata=metadata)


def add_assistant_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to add an assistant message.
    
    Args:
        content: Message content
        metadata: Optional metadata
        
    Returns:
        True if message was added successfully
    """
    mcp = get_mcp()
    return mcp.add_message(MessageRole.ASSISTANT, content, metadata=metadata)


def get_chat_history(last_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to get chat history.
    
    Args:
        last_n: Last N messages to return
        
    Returns:
        List of message dictionaries
    """
    mcp = get_mcp()
    return mcp.get_conversation_history(last_n_messages=last_n) 