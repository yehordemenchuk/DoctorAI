"""
© 2025 DoctorAI. All rights reserved. 
Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation.

Professional RAG (Retrieval-Augmented Generation) Implementation for Medical Document Processing
"""

import os
import logging
from typing import List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from app.config import Config

# Configure logging
logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Custom exception for RAG-related errors."""
    pass


class MedicalRAGSystem:
    """
    Professional Medical Retrieval-Augmented Generation System.
    
    This class provides a comprehensive RAG implementation specifically designed
    for medical document processing and retrieval.
    """
    
    def __init__(
        self, 
        model_name: str = None,
        pdf_folder: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the Medical RAG System.
        
        Args:
            model_name: Name of the embedding model to use
            pdf_folder: Path to folder containing PDF documents
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.model_name = model_name or Config.TRANSFORMER_EMBEDDINGS_MODEL_NAME
        self.pdf_folder = Path(pdf_folder or getattr(Config, 'PDF_FOLDER', './library'))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.text_chunks: List[str] = []
        
        logger.info(f"Initialized MedicalRAGSystem with model: {self.model_name}")
    
    def _load_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        Load the HuggingFace embedding model.
        
        Returns:
            Configured HuggingFace embedding model
            
        Raises:
            RAGError: If model loading fails
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RAGError(f"Failed to load embedding model: {str(e)}")
    
    def _load_single_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a single PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects from the PDF
            
        Raises:
            RAGError: If PDF loading fails
        """
        try:
            logger.debug(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(str(file_path))
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {str(e)}")
            raise RAGError(f"Failed to load PDF {file_path}: {str(e)}")
    
    def load_documents(self, max_workers: int = 4) -> List[Document]:
        """
        Load all PDF documents from the specified folder.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            
        Returns:
            List of all loaded documents
            
        Raises:
            RAGError: If document loading fails
        """
        if not self.pdf_folder.exists():
            raise RAGError(f"PDF folder does not exist: {self.pdf_folder}")
        
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_folder}")
            return []
        
        logger.info(f"Loading {len(pdf_files)} PDF files from {self.pdf_folder}")
        documents = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._load_single_pdf, pdf_file): pdf_file
                for pdf_file in pdf_files
            }
            
            for future in as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    docs = future.result()
                    documents.extend(docs)
                    logger.debug(f"Successfully loaded {len(docs)} pages from {pdf_file.name}")
                except Exception as e:
                    logger.error(f"Error loading {pdf_file}: {str(e)}")
        
        self.documents = documents
        logger.info(f"Successfully loaded {len(documents)} document pages")
        return documents
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + self.chunk_size // 2:
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
                
        return chunks
    
    def create_embeddings(self) -> np.ndarray:
        """
        Create embeddings for all document chunks.
        
        Returns:
            Numpy array of embeddings
            
        Raises:
            RAGError: If embedding creation fails
        """
        if not self.documents:
            raise RAGError("No documents loaded. Call load_documents() first.")
        
        if self.embedding_model is None:
            self.embedding_model = self._load_embedding_model()
        
        # Extract and chunk text from documents
        all_chunks = []
        for doc in self.documents:
            chunks = self._chunk_text(doc.page_content)
            all_chunks.extend(chunks)
        
        self.text_chunks = all_chunks
        logger.info(f"Created {len(all_chunks)} text chunks")
        
        try:
            logger.info("Creating embeddings for text chunks...")
            embeddings = []
            
            # Process embeddings in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                batch_embeddings = self.embedding_model.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                
                if i % (batch_size * 10) == 0:
                    logger.debug(f"Processed {i}/{len(all_chunks)} chunks")
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
            raise RAGError(f"Failed to create embeddings: {str(e)}")
    
    def build_index(self, embeddings: Optional[np.ndarray] = None) -> faiss.Index:
        """
        Build FAISS index for similarity search.
        
        Args:
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            FAISS index
            
        Raises:
            RAGError: If index building fails
        """
        if embeddings is None:
            embeddings = self.create_embeddings()
        
        try:
            logger.info("Building FAISS index...")
            dimension = embeddings.shape[1]
            
            # Use IndexFlatIP for cosine similarity (with normalized embeddings)
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            self.index = index
            logger.info(f"Built FAISS index with {index.ntotal} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {str(e)}")
            raise RAGError(f"Failed to build FAISS index: {str(e)}")
    
    def search_similar_documents(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for similar documents using the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of tuples (text_chunk, similarity_score)
            
        Raises:
            RAGError: If search fails
        """
        if self.index is None:
            raise RAGError("Index not built. Call build_index() first.")
        
        if self.embedding_model is None:
            self.embedding_model = self._load_embedding_model()
        
        try:
            logger.debug(f"Searching for query: {query[:100]}...")
            
            # Create query embedding
            query_embedding = np.array(
                [self.embedding_model.embed_query(query)], 
                dtype=np.float32
            )
            
            # Search in index
            similarities, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= similarity_threshold and idx < len(self.text_chunks):
                    results.append((self.text_chunks[idx], float(similarity)))
            
            logger.debug(f"Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise RAGError(f"Search failed: {str(e)}")
    
    def get_context_for_query(
        self, 
        query: str, 
        max_context_length: int = 4000,
        top_k: int = 5
    ) -> str:
        """
        Get relevant context for a query, formatted for LLM consumption.
        
        Args:
            query: Search query
            max_context_length: Maximum length of context to return
            top_k: Number of chunks to consider
            
        Returns:
            Formatted context string
        """
        try:
            results = self.search_similar_documents(query, top_k=top_k)
            
            if not results:
                logger.warning("No relevant documents found for query")
                return "No relevant medical documents found for this query."
            
            context_parts = []
            current_length = 0
            
            for i, (text, score) in enumerate(results):
                header = f"--- Medical Document Excerpt {i+1} (Relevance: {score:.3f}) ---\n"
                chunk = f"{header}{text}\n\n"
                
                if current_length + len(chunk) > max_context_length:
                    break
                
                context_parts.append(chunk)
                current_length += len(chunk)
            
            context = "".join(context_parts)
            logger.debug(f"Generated context of length {len(context)}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context: {str(e)}")
            return f"Error retrieving medical information: {str(e)}"
    
    def initialize_system(self) -> bool:
        """
        Initialize the complete RAG system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Medical RAG System...")
            
            # Load documents
            documents = self.load_documents()
            if not documents:
                logger.warning("No documents loaded")
                return False
            
            # Build index
            self.build_index()
            
            logger.info("Medical RAG System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            return False


# Global instance for easy access
_rag_system: Optional[MedicalRAGSystem] = None


def get_rag_system() -> MedicalRAGSystem:
    """
    Get or create the global RAG system instance.
    
    Returns:
        MedicalRAGSystem instance
    """
    global _rag_system
    
    if _rag_system is None:
        _rag_system = MedicalRAGSystem()
        _rag_system.initialize_system()
    
    return _rag_system


def search_medical_documents(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Convenience function to search medical documents.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        List of (text, similarity_score) tuples
    """
    rag_system = get_rag_system()
    return rag_system.search_similar_documents(query, top_k=top_k)


def get_medical_context(query: str, max_length: int = 4000) -> str:
    """
    Convenience function to get formatted medical context.
    
    Args:
        query: Search query
        max_length: Maximum context length
        
    Returns:
        Formatted context string
    """
    rag_system = get_rag_system()
    return rag_system.get_context_for_query(query, max_context_length=max_length)



