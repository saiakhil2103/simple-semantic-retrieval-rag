import os 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma # Imported Chroma instead of Qdrant

from loaders import load_documents
from chunking import chunk_documents

load_dotenv()

os.environ["OPENAI_API_KEY"] = "paste the secret key here"

COLLECTION_NAME = 'arxiv_papers'
CHROMA_PATH = 'chroma_data' # Updated path name for clarity

def build_dense_index():
    docs = load_documents()
    chunks = chunk_documents(docs)

    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    # Chroma handles collection creation automatically if it doesn't exist.
    # It also automatically saves data locally to the persist_directory.
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    vectorstore.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks into Chroma collection '{COLLECTION_NAME}'")
    return vectorstore

if __name__ == '__main__':
    build_dense_index()