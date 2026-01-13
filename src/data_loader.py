from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_and_split_documents(self) -> List[Document]:
        raw_documents: List[Document] = []

        for file_path in self.data_dir.iterdir():
            if file_path.suffix.lower() == ".pdf":
                loader = PyMuPDFLoader(str(file_path))
                docs = loader.load()
                raw_documents.extend(docs)

            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                raw_documents.extend(docs)

        # 🔹 Chunking happens here
        split_documents = self.text_splitter.split_documents(raw_documents)

        # Add metadata
        for doc in split_documents:
            doc.metadata["source"] = doc.metadata.get("source", "unknown")

        return split_documents
