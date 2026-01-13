"""
Test script for chunking the PDF file
"""
import sys
sys.path.insert(0, ".")

from src.data_loader import DataLoader

def test_pdf_chunking():
    print("=" * 60)
    print("Testing PDF Chunking")
    print("=" * 60)
    
    # Initialize DataLoader
    loader = DataLoader("data")
    print(f"✓ DataLoader initialized with path: {loader.data_dir}")
    
    # Load and split documents
    print("\nLoading and splitting PDF documents...")
    documents = loader.load_and_split_documents()
    
    print(f"\n✓ Total chunks created: {len(documents)}")
    
    # Show statistics
    if documents:
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chars = total_chars // len(documents)
        
        print(f"✓ Total characters: {total_chars:,}")
        print(f"✓ Average chunk size: {avg_chars} chars")
        
        # Show first 3 chunks preview
        print("\n" + "-" * 60)
        print("SAMPLE CHUNKS PREVIEW:")
        print("-" * 60)
        
        for i, doc in enumerate(documents[:3], 1):
            print(f"\n📄 Chunk {i}:")
            print(f"   Source: {doc.metadata.get('source', 'unknown')}")
            if 'page' in doc.metadata:
                print(f"   Page: {doc.metadata.get('page')}")
            print(f"   Length: {len(doc.page_content)} chars")
            print(f"   Preview: {doc.page_content[:150]}...")
        
        # Show unique metadata keys
        print("\n" + "-" * 60)
        print("METADATA KEYS FOUND:")
        all_keys = set()
        for doc in documents:
            all_keys.update(doc.metadata.keys())
        for key in sorted(all_keys):
            print(f"   • {key}")
    
    print("\n" + "=" * 60)
    print("✅ PDF Chunking test COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_pdf_chunking()
