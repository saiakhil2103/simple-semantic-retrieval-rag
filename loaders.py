from langchain_community.document_loaders import PyPDFLoader
import os 

def load_documents(data_dir='data'):
    documents = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.pdf'):
            path = os.path.join(data_dir, fname)
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata['source'] = fname
            documents.extend(docs)
    return documents 

if __name__ == '__main__':
    docs = load_documents()
    print(f'Loaded {len(docs)} pages from PDFs')
    print(docs[0].page_content[:300])
    print(docs[0].metadata)