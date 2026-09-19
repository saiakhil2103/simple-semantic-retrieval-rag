import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated from Qdrant

load_dotenv()

COLLECTION_NAME = "arxiv_papers"
CHROMA_PATH = "chroma_data"  # Matching your indexing directory

os.environ["OPENAI_API_KEY"] = "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"

def get_dense_retriever():
    """
    Connects to the local Chroma vector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Chroma handles loading from the local persistence directory
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return vectorstore

def dense_search(query, k=10):
    """
    Performs a vector search and returns documents with distance scores.
    """
    vectorstore = get_dense_retriever()
    
    # Returns a list of tuples: (Document, score)
    # Note: For Chroma's default distance metric, a lower score means higher similarity.
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

if __name__ == "__main__":
    hits = dense_search("how does re-ranking improve retrieval quality", k=5)
    
    for doc, score in hits:
        # We use .get() to avoid KeyError depending on metadata capitalization 
        title = doc.metadata.get('Title', doc.metadata.get('title', 'Unknown Title'))
        page = doc.metadata.get('Page', doc.metadata.get('page', 'N/A'))
        
        print(f"[{score:.4f}] {title[:60]} (page {page})")
        print(doc.page_content[:150].replace("\n", " "))
        print("---")