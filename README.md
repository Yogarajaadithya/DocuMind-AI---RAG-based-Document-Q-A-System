# DocuMind AI – RAG-based Document Q&A System

A modular Retrieval-Augmented Generation (RAG) system for intelligent document question-answering, built with Python.

## 📁 Project Structure

```
rag_modular/
├── data/                   # Store your documents here
├── src/
│   ├── __init__.py         # Package initialization
│   ├── data_loader.py      # Document loading and preprocessing
│   ├── embedding.py        # Text embedding generation
│   ├── vector_store.py     # Vector database operations
│   ├── search.py           # Semantic search functionality
│   └── app.py              # Main application
├── .env                    # Environment variables (API keys, configs)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. **Clone/Navigate to the project:**
   ```bash
   cd rag_modular
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   # Copy the example file and add your API keys
   cp .env.example .env
   # Edit .env file with your actual Azure OpenAI credentials
   ```

## 📖 Usage

### Basic Usage

```python
from src.app import RAGApplication

# Initialize the application
app = RAGApplication()

# Ingest documents
app.ingest_documents("data/")

# Query the system
response = app.query("What is the main topic?")
print(response)
```

### Module-by-Module Usage

```python
from src.data_loader import DataLoader
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.search import SearchEngine

# Load documents
loader = DataLoader("data/")
documents = loader.load_text_files()
chunks = loader.chunk_documents(documents)

# Generate embeddings
embedder = EmbeddingModel()
embeddings = embedder.embed_batch([c['content'] for c in chunks])

# Store in vector database
store = VectorStore()
store.add_documents(documents=[c['content'] for c in chunks], embeddings=embeddings)

# Search
search = SearchEngine(embedder, store)
results = search.search("your query here")
```

## 🔧 Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | - |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Azure OpenAI deployment name | - |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version | `2024-02-15-preview` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Document chunk size | `500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |

## 🧩 Modules

### DataLoader (`data_loader.py`)
- Load documents from various formats (TXT, PDF)
- Chunk documents for processing

### EmbeddingModel (`embedding.py`)
- Generate text embeddings using sentence transformers
- Support for batch processing

### VectorStore (`vector_store.py`)
- Store embeddings in a vector database (ChromaDB)
- Similarity search capabilities

### SearchEngine (`search.py`)
- Semantic search over stored documents
- Context retrieval for LLM

### RAGApplication (`app.py`)
- Main orchestration class
- End-to-end RAG pipeline

## 📝 TODO

- [ ] Implement document loading for various formats
- [ ] Add LLM integration for response generation
- [ ] Create REST API with FastAPI
- [ ] Add Streamlit UI
- [ ] Implement caching for embeddings
- [ ] Add evaluation metrics

## 📄 License

MIT License
