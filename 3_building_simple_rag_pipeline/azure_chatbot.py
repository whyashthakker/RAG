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
import time
import random
import tempfile
from langchain_community.document_loaders import BSHTMLLoader

from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings

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

def scrape_website(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text from various elements
        content = []
        for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']):
            if elem.text.strip():
                content.append(elem.text.strip())
        
        # If no content found, try to get all text from body
        if not content:
            body = soup.find('body')
            if body:
                content = [body.get_text(separator='\n', strip=True)]
        
        if not content:
            print("Warning: No content found. The website might have unusual structure or require JavaScript.")
            return []
        
        return content
    except requests.RequestException as e:
        print(f"Error scraping the website: {e}")
        return []

def clean_content(content_list):
    # Remove very short or common unwanted items
    cleaned = [text for text in content_list if len(text) > 20 and not any(item in text.lower() for item in ['sign up', 'sign in', 'cookie', 'privacy policy'])]
    return cleaned

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
    
    # Use a temporary file to store the HTML content
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    try:
        # Try to use BSHTMLLoader with default settings (which uses 'lxml')
        loader = BSHTMLLoader(temp_file_path)
        documents = loader.load()
    except ImportError:
        print("'lxml' is not installed. Falling back to built-in 'html.parser'.")
        # If 'lxml' is not available, use the built-in 'html.parser'
        loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
        documents = loader.load()

    # Clean up the temporary file
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
llm = AzureOpenAI(
    deployment_name="your-deployment-name",  # Replace with your actual deployment name
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

def rag_pipeline(query, qa_chain, vectorstore):
    relevant_docs = vectorstore.similarity_search_with_score(query, k=3)
    
    print("\nTop 3 most relevant chunks:")
    context = ""
    for i, (doc, score) in enumerate(relevant_docs, 1):
        print(f"{i}. Relevance Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:200]}...")
        print()
        context += doc.page_content + "\n\n"

    # Print the full prompt
    full_prompt = PROMPT.format(context=context, question=query)
    print("\nFull Prompt sent to the model:")
    print(full_prompt)
    print("\n" + "="*50 + "\n")

    response = qa_chain.invoke({"query": query})
    return response['result']

if __name__ == "__main__":
    print("Welcome to the Enhanced Web Scraping RAG Pipeline.")
    
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
            
            print("Creating embeddings and vector store...")

            embeddings = AzureOpenAIEmbeddings(
                deployment="your-embedding-deployment-name", 
                model="text-embedding-ada-002"
            )
                        
            print_sample_embeddings(texts, embeddings)
            
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            print("\nRAG Pipeline initialized. You can now enter your queries.")
            print("Enter 'new' to query a new website or 'quit' to exit the program.")
            
            while True:
                user_query = input("\nEnter your query: ")
                if user_query.lower() == 'quit':
                    print("Exiting the program. Goodbye!")
                    exit()
                elif user_query.lower() == 'new':
                    break
                
                result = rag_pipeline(user_query, qa, vectorstore)
                print(f"RAG Response: {result}")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try a different URL or check your internet connection.")