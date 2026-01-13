"""
RAG Modular - A modular Retrieval-Augmented Generation system.
"""

from .data_loader import DataLoader
from .embedding import EmbeddingModel
from .vector_store import VectorStore
from .search import SearchEngine

__version__ = "0.1.0"
__all__ = ["DataLoader", "EmbeddingModel", "VectorStore", "SearchEngine"]
