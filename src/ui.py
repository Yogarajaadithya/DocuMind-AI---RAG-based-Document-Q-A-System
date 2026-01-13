import os
import streamlit as st
from dotenv import load_dotenv

from data_loader import DataLoader
from embedding import EmbeddingPipeline
from vector_store import VectorStore
from search import SearchEngine


@st.cache_resource
def initialize_rag():
    load_dotenv()

    # Use correct path relative to src folder
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    index_path = os.path.join(os.path.dirname(__file__), "..", "faiss_index")

    # Load & chunk documents
    loader = DataLoader(data_path)
    documents = loader.load_and_split_documents()

    # Embeddings
    embedder = EmbeddingPipeline()

    # Vector store
    vector_store = VectorStore(
        embedding_model=embedder.embedding_model,
        index_path=index_path
    )

    if os.path.exists(index_path):
        vector_store.load_index()
    else:
        vector_store.build_index(documents)
        vector_store.save_index()

    # Search engine
    return SearchEngine(vector_store)


def main():
    st.set_page_config(
        page_title="RAG Document Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for ChatGPT-like interface
    st.markdown("""
    <style>
    /* Dark theme */
    .stApp {
        background-color: #343541;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Message styling */
    .user-message {
        background-color: #343541;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        display: flex;
        gap: 15px;
    }
    
    .assistant-message {
        background-color: #444654;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        display: flex;
        gap: 15px;
    }
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #5436DA 0%, #7B68EE 100%);
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, #10a37f 0%, #1ED760 100%);
    }
    
    .message-content {
        color: #ECECF1;
        line-height: 1.6;
        flex-grow: 1;
    }
    
    /* Header */
    .header {
        text-align: center;
        padding: 40px 20px;
        color: #ECECF1;
    }
    
    .header h1 {
        font-size: 2.5rem;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #10a37f, #1ED760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header p {
        color: #8e8ea0;
        font-size: 1.1rem;
    }
    
    /* Input area */
    .stTextInput > div > div > input {
        background-color: #40414f;
        border: 1px solid #565869;
        color: white;
        border-radius: 12px;
        padding: 15px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10a37f;
        box-shadow: 0 0 0 1px #10a37f;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #10a37f 0%, #1a7f64 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a7f64 0%, #10a37f 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(16, 163, 127, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #202123;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #10a37f !important;
    }
    
    /* Welcome cards */
    .welcome-card {
        background-color: #40414f;
        border: 1px solid #565869;
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .welcome-card:hover {
        border-color: #10a37f;
        transform: translateY(-3px);
    }
    
    .welcome-card h4 {
        color: #ECECF1;
        margin-bottom: 8px;
    }
    
    .welcome-card p {
        color: #8e8ea0;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False

    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 RAG Assistant")
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("##### 📄 Document Loaded")
        st.markdown("Master Assessments PDF")
        
        st.markdown("---")
        st.markdown("##### ⚙️ Settings")
        num_chunks = st.slider("Context chunks (k)", 1, 10, 3)
        
        st.markdown("---")
        st.markdown(
            "<p style='color: #8e8ea0; font-size: 0.8rem;'>Built with LangChain + FAISS + Azure OpenAI</p>",
            unsafe_allow_html=True
        )

    # Initialize RAG
    with st.spinner("🚀 Initializing RAG system..."):
        search_engine = initialize_rag()
        st.session_state.rag_initialized = True

    # Main chat area
    if not st.session_state.messages:
        # Welcome screen
        st.markdown("""
        <div class="header">
            <h1>🤖 RAG Document Assistant</h1>
            <p>Ask questions about your documents and get AI-powered answers</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example questions
        st.markdown("##### 💡 Try asking:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 What is this document about?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "What is this document about?"})
                st.rerun()
        
        with col2:
            if st.button("🔍 What are the main topics?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "What are the main topics covered in this document?"})
                st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="avatar user-avatar">👤</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <div class="avatar bot-avatar">🤖</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Ask a question about your document...",
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send ➤", use_container_width=True)

    # Process input
    if send_button and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                answer = search_engine.ask(user_input, k=num_chunks)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ Error: {str(e)}"
                })
        
        st.rerun()


if __name__ == "__main__":
    main()
