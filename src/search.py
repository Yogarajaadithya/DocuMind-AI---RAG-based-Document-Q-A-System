import os
from typing import List

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class SearchEngine:
    """
    Handles retrieval-augmented generation using
    FAISS retrieval + Azure OpenAI GPT-4.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.2
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an assistant that answers questions strictly using the provided context.
If the answer is not present in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer concisely and clearly.
"""
        )

    def _build_context(self, documents: List[Document]) -> str:
        """
        Combines retrieved document chunks into a single context string.
        """
        return "\n\n".join(
            f"(Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}"
            for doc in documents
        )

    def ask(self, question: str, k: int = 3) -> str:
        """
        Executes full RAG pipeline:
        retrieval → augmentation → generation
        """
        retrieved_docs = self.vector_store.similarity_search(question, k=k)

        context = self._build_context(retrieved_docs)

        messages = self.prompt.format_messages(
            context=context,
            question=question
        )

        response = self.llm.invoke(messages)
        return response.content
