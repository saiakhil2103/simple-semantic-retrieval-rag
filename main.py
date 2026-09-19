import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Assuming you saved the retrieval code from the previous step in a file named `retrieval.py`
from retriever import dense_search 

load_dotenv()

os.environ["OPENAI_API_KEY"] = "paste the secret key here"

# 1. Define the Prompt Template
# We strictly instruct the LLM to use the provided context and cite its sources.
RAG_PROMPT_TEMPLATE = """You are a helpful research assistant. Answer the question based ONLY on the provided research paper context below.
If the context does not contain enough information to answer the question, clearly state that you don't know based on the provided papers.
Always cite the source paper title (and page number if available).

Context:
{context}

Question: {question}

Answer:"""

def format_context(hits):
    """
    Formats the retrieved Qdrant/Chroma hits into a single string for the prompt.
    """
    formatted_docs = []
    for doc, score in hits:
        # Safely extract metadata regardless of capitalization
        title = doc.metadata.get('Title', doc.metadata.get('title', 'Unknown Title'))
        page = doc.metadata.get('Page', doc.metadata.get('page', 'N/A'))
        
        doc_str = f"Title: {title} (Page: {page}, Score: {score:.4f})\nContent:\n{doc.page_content}"
        formatted_docs.append(doc_str)
        
    return "\n\n---\n\n".join(formatted_docs)

def generate_answer(query: str, k: int = 5):
    """
    Retrieves context and streams the LLM response.
    """
    print(f"\n[1] Searching vector database for top {k} chunks...")
    hits = dense_search(query, k=k)
    
    if not hits:
        print("No relevant documents found.")
        return
        
    print("[2] Formatting context and calling LLM...\n")
    context_text = format_context(hits)
    
    # 2. Setup LLM and Chain using LCEL (LangChain Expression Language)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Using gpt-4o-mini for cost-efficiency
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Connect the prompt, model, and output parser together
    chain = prompt | llm | StrOutputParser()
    
    # 3. Stream the response to the terminal for a ChatGPT-like feel
    print("================ ANSWER ================")
    for chunk in chain.stream({"context": context_text, "question": query}):
        print(chunk, end="", flush=True)
    print("\n========================================\n")

if __name__ == "__main__":
    print("Welcome to the arXiv RAG Chatbot! (Type 'exit' or 'quit' to stop)")
    
    # Interactive CLI Loop
    while True:
        user_query = input("\nAsk a question about your papers: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_query.strip():
            continue
            
        generate_answer(user_query, k=5)


# user_query -> where does acchint live? 
# dense_search -> [doc1, doc2, doc3, doc4, doc5]
# format_context ->
# for loop  
# [
#     "Title: {doc1.title} Page: {doc1.page} content: {doc1.content}",
#     "Title: {doc2.title} Page: {doc2.page} content: {doc2.content}"
# ]

# "-".join(["a", "b", "c", "d", "e"]) -> a-b-c-d-e 

# "\n\n---\n\n"

# """
# "Title: {doc1.title} Page: {doc1.page} content: {doc1.content}"

# --- 

# "Title: {doc1.title} Page: {doc1.page} content: {doc1.content}",

# ---

# "Title: {doc1.title} Page: {doc1.page} content: {doc1.content}",

# --- 

# "Title: {doc1.title} Page: {doc1.page} content: {doc1.content}",

# ---

# "Title: {doc1.title} Page: {doc1.page} content: {doc1.content}",

# """


# You are a helpful research assistant. Answer the question based ONLY on the provided research paper context below.
# If the context does not contain enough information to answer the question, clearly state that you don't know based on the provided papers.
# Always cite the source paper title (and page number if available).

# Context:
# {context}

# Question: {question}

# Answer:


# 1. query from user - query enhancement 
# 2. retriever - fetch the relevant context - bm25 keyword - hybrid -> verify the documents
# 3. list of documents - fusion techniques 
# 4. final list of documents - reranking llm - (query + document - reranked score)
# 3. prompt augmentation -
# 4. llm - 
# 5. response -   


# 1. retrieval eval  -> recall@k precision@k 
# 2. generation eval 
# ragas 
# llm as a judge 

# 2 hours course 