from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '. ', ' ', ''],
        length_function=len
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['title'] = chunk.metadata.get('title', 'unknown')
        chunk.metadata['source'] = chunk.metadata.get('source', 'unknown')
        chunk.metadata['page'] = chunk.metadata.get('page', None)
    return chunks

if __name__ == '__main__':
    from loaders import load_documents 
    docs = load_documents() 
    chunks = chunk_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    print(chunks[0].page_content[:300])
    print(chunks[0].metadata)