"""
Full RAG Pipeline Test - Tests all components end-to-end
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

def test_full_pipeline():
    print("=" * 70)
    print("🚀 FULL RAG PIPELINE TEST")
    print("=" * 70)
    
    # Load environment
    load_dotenv()
    
    # ============================================
    # STEP 1: Data Loader
    # ============================================
    print("\n📁 STEP 1: Testing DataLoader...")
    print("-" * 50)
    
    from data_loader import DataLoader
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    loader = DataLoader(data_path)
    documents = loader.load_and_split_documents()
    
    print(f"✅ Loaded {len(documents)} document chunks")
    print(f"   Sample chunk: {documents[0].page_content[:100]}...")
    
    # ============================================
    # STEP 2: Embedding Pipeline
    # ============================================
    print("\n🧠 STEP 2: Testing EmbeddingPipeline...")
    print("-" * 50)
    
    from embedding import EmbeddingPipeline
    
    embedder = EmbeddingPipeline()
    
    # Test single query embedding
    test_query = "What is this document about?"
    query_embedding = embedder.embed_query(test_query)
    
    print(f"✅ Embedding model loaded: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   Test query: '{test_query}'")
    print(f"   Embedding dimension: {len(query_embedding)}")
    print(f"   Sample values: [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ...]")
    
    # ============================================
    # STEP 3: Vector Store (FAISS)
    # ============================================
    print("\n📦 STEP 3: Testing VectorStore (FAISS)...")
    print("-" * 50)
    
    from vector_store import VectorStore
    
    index_path = os.path.join(os.path.dirname(__file__), "..", "faiss_index_test")
    
    vector_store = VectorStore(
        embedding_model=embedder.embedding_model,
        index_path=index_path
    )
    
    # Build fresh index for testing
    print("   Building FAISS index from documents...")
    vector_store.build_index(documents)
    print(f"✅ FAISS index built with {len(documents)} vectors")
    
    # Test similarity search
    test_results = vector_store.similarity_search("What is choice?", k=2)
    print(f"   Test search for 'What is choice?':")
    for i, doc in enumerate(test_results, 1):
        print(f"   [{i}] {doc.page_content[:80]}...")
    
    # Save index
    vector_store.save_index()
    print(f"✅ Index saved to: {index_path}")
    
    # ============================================
    # STEP 4: Search Engine (RAG with Azure GPT-4)
    # ============================================
    print("\n🔍 STEP 4: Testing SearchEngine (RAG)...")
    print("-" * 50)
    
    from search import SearchEngine
    
    search_engine = SearchEngine(vector_store)
    print("✅ SearchEngine initialized with Azure OpenAI GPT-4")
    
    # Check Azure credentials
    print(f"   Azure Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'NOT SET')}")
    print(f"   Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'NOT SET')}")
    
    # ============================================
    # STEP 5: Full RAG Query
    # ============================================
    print("\n💬 STEP 5: Testing Full RAG Query...")
    print("-" * 50)
    
    test_question = "What is this document about?"
    print(f"   Question: {test_question}")
    print("   Generating answer...")
    
    try:
        answer = search_engine.ask(test_question, k=3)
        print(f"\n✅ RAG ANSWER:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
    except Exception as e:
        print(f"❌ Error during RAG query: {e}")
        raise
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED! RAG PIPELINE IS FULLY OPERATIONAL!")
    print("=" * 70)
    print("""
    ✅ DataLoader      - PDF loading and chunking works
    ✅ EmbeddingPipeline - Text to vector conversion works  
    ✅ VectorStore     - FAISS indexing and search works
    ✅ SearchEngine    - Azure GPT-4 RAG generation works
    
    🚀 Your RAG system is ready to use!
    """)
    
    # Cleanup test index
    import shutil
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        print(f"🧹 Cleaned up test index: {index_path}")


if __name__ == "__main__":
    test_full_pipeline()
