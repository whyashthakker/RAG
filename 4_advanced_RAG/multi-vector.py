import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import numpy as np
import tempfile
from langchain_community.document_loaders import BSHTMLLoader
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuration variables
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS = 15000
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.4

# Set up OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None

def process_website(url):
    html_content = fetch_html(url)
    if not html_content:
        raise ValueError("No content could be fetched from the website.")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    try:
        loader = BSHTMLLoader(temp_file_path)
        documents = loader.load()
    except ImportError:
        print("'lxml' is not installed. Falling back to built-in 'html.parser'.")
        loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
        documents = loader.load()

    os.unlink(temp_file_path)

    print(f"\nNumber of documents loaded: {len(documents)}")
    if documents:
        print("Sample of loaded content:")
        print(documents[0].page_content[:200] + "...")
        print(f"Metadata: {documents[0].metadata}")
    
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    print(f"Number of text chunks after splitting: {len(texts)}")
    return texts

def print_sample_embeddings(texts, embeddings):
    if texts:
        sample_text = texts[0].page_content
        sample_embedding = embeddings.embed_query(sample_text)
        print("\nSample Text:")
        print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
        print("\nSample Embedding (first 10 dimensions):")
        print(np.array(sample_embedding[:10]))
        print(f"\nEmbedding shape: {np.array(sample_embedding).shape}")
    else:
        print("No texts available for embedding sample.")

# Set up OpenAI language model
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)

# Set up the retrieval-based QA system with a simplified prompt template
template = """Context: {context}

Question: {question}

Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

But, if the question is generic, then go ahead and answer the question, example what is a electric vehicle?
"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# NEW: Function to create summaries for multivector retrieval
def create_summaries(texts):
    summarize_prompt = PromptTemplate.from_template("Summarize the following text:\n\n{text}")
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    summaries = summarize_chain.batch([{"text": doc.page_content} for doc in texts], {"max_concurrency": 5})
    return summaries

# NEW: Function to set up multivector retrieval
def setup_multivector_retrieval(texts, embeddings):
    summaries = create_summaries(texts)
    
    # Create and populate the vector stores
    vectorstore_full = FAISS.from_documents(texts, embeddings)
    vectorstore_summaries = FAISS.from_texts(summaries, embeddings)
    
    # Create the storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"
    
    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore_summaries,
        docstore=store,
        id_key=id_key,
    )
    
    # Add the documents to the retriever
    doc_ids = [str(i) for i in range(len(texts))]
    retriever.docstore.mset(list(zip(doc_ids, texts)))
    retriever.vectorstore.add_documents([
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ])
    
    return retriever, vectorstore_full

def rag_pipeline(query, qa_chain, retriever, vectorstore_full):
    # NEW: Perform hybrid search
    relevant_docs_summaries = retriever.get_relevant_documents(query)
    relevant_docs_full = vectorstore_full.similarity_search_with_score(query, k=3)
    
    print("\nTop 3 most relevant chunks from summary-based retrieval:")
    context_summaries = ""
    for i, doc in enumerate(relevant_docs_summaries[:3], 1):
        print(f"{i}. Content: {doc.page_content[:200]}...")
        print()
        context_summaries += doc.page_content + "\n\n"
    
    print("\nTop 3 most relevant chunks from full-text retrieval:")
    context_full = ""
    for i, (doc, score) in enumerate(relevant_docs_full, 1):
        print(f"{i}. Relevance Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:200]}...")
        print()
        context_full += doc.page_content + "\n\n"
    
    # Combine contexts from both retrieval methods
    combined_context = context_summaries + "\n" + context_full
    
    # Print the full prompt
    full_prompt = PROMPT.format(context=combined_context, question=query)
    print("\nFull Prompt sent to the model:")
    print(full_prompt)
    print("\n" + "="*50 + "\n")

    response = qa_chain.invoke({"query": query})
    return response['result']

if __name__ == "__main__":
    print("Welcome to the Enhanced Web Scraping RAG Pipeline with Multivector and Hybrid Search.")
    
    while True:
        url = input("Please enter the URL of the website you want to query (or 'quit' to exit): ")
        if url.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break
        
        try:
            print("Processing website content...")
            texts = process_website(url)
            
            if not texts:
                print("No content found on the website. Please try a different URL.")
                continue
            
            print("Creating embeddings and vector stores...")
            embeddings = OpenAIEmbeddings()
            
            print_sample_embeddings(texts, embeddings)
            
            # NEW: Set up multivector retrieval
            multivector_retriever, vectorstore_full = setup_multivector_retrieval(texts, embeddings)
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=multivector_retriever,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            print("\nRAG Pipeline initialized with Multivector and Hybrid Search. You can now enter your queries.")
            print("Enter 'new' to query a new website or 'quit' to exit the program.")
            
            while True:
                user_query = input("\nEnter your query: ")
                if user_query.lower() == 'quit':
                    print("Exiting the program. Goodbye!")
                    exit()
                elif user_query.lower() == 'new':
                    break
                
                result = rag_pipeline(user_query, qa, multivector_retriever, vectorstore_full)
                print(f"RAG Response: {result}")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try a different URL or check your internet connection.")