import os
from dotenv import load_dotenv

from data_loader import DataLoader
from embedding import EmbeddingPipeline
from vector_store import VectorStore
from search import SearchEngine


def main():
    # Load environment variables
    load_dotenv()

    # 1. Load & chunk documents
    loader = DataLoader("data")
    documents = loader.load_and_split_documents()

    # 2. Initialize embedding pipeline
    embedder = EmbeddingPipeline()

    # 3. Initialize vector store
    vector_store = VectorStore(
        embedding_model=embedder.embedding_model,
        index_path="faiss_index"
    )

    # 4. Build or load FAISS index
    if os.path.exists("faiss_index"):
        print("Loading existing FAISS index...")
        vector_store.load_index()
    else:
        print("Building FAISS index...")
        vector_store.build_index(documents)
        vector_store.save_index()

    # 5. Initialize search engine (RAG)
    search_engine = SearchEngine(vector_store)

    # 6. Ask questions in a loop
    print("\nRAG system ready. Ask questions (type 'exit' to quit).\n")

    while True:
        question = input("Question: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = search_engine.ask(question)
        print("\nAnswer:")
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
