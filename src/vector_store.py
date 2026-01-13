from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
import os


class VectorStore:
    """
    Handles vector storage, persistence, and similarity search using FAISS.
    """

    def __init__(self, embedding_model, index_path: str = "faiss_index"):
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.vector_store = None

    def build_index(self, documents: List[Document]):
        """
        Creates a FAISS index from documents and embeddings.
        """
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

    def save_index(self):
        """
        Saves FAISS index to disk.
        """
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Build index first.")

        self.vector_store.save_local(self.index_path)

    def load_index(self):
        """
        Loads FAISS index from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("FAISS index not found on disk.")

        self.vector_store = FAISS.load_local(
            self.index_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def similarity_search(self, query: str, k: int = 3):
        """
        Searches for top-k most similar document chunks.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")

        return self.vector_store.similarity_search(query, k=k)
