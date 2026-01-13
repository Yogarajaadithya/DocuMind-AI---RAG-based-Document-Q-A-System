from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingPipeline:
    """
    Handles text-to-vector embedding using a local open-source model.
    """

    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, documents):
        """
        Converts document chunks into embedding vectors.
        """
        return self.embedding_model.embed_documents(
            [doc.page_content for doc in documents]
        )

    def embed_query(self, query: str):
        """
        Converts a user query into an embedding vector.
        """
        return self.embedding_model.embed_query(query)
