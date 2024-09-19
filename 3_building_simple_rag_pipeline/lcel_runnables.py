import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import tempfile

# Import LCEL Runnables
from langchain.runnables import (
    RunnableLambda,
    RunnableSequence,
    RunnableParallel,
    RunnableRetry,
    RunnableFallbacks,
    RunnableConfig,
    ConfigurableField,
    RunnablePassthrough,
)

# Import LangChain components
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import BSHTMLLoader

# Configuration variables
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS = 1500  # Reduced for practical purposes
MODEL_NAME = "gpt-3.5-turbo"  # Adjusted to a commonly accessible model
TEMPERATURE = 0.4

# Set up OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Function to scrape website content
def fetch_html(url):
    print(f"Fetching HTML content from {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print("Successfully fetched HTML content.")
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None

# Function to process HTML content into documents
def process_website(html_content):
    print("Processing HTML content into documents.")
    if not html_content:
        raise ValueError("No content could be fetched from the website.")

    # Use a temporary file to store the HTML content
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    try:
        # Use BSHTMLLoader to parse HTML
        loader = BSHTMLLoader(temp_file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from HTML.")
    except ImportError:
        print("'lxml' is not installed. Falling back to built-in 'html.parser'.")
        loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
        documents = loader.load()

    # Clean up the temporary file
    os.unlink(temp_file_path)

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} text chunks.")
    return texts

# Function to print sample embeddings
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

# Function for the RAG pipeline
def rag_pipeline(inputs):
    query = inputs['query']
    qa_chain = inputs['qa_chain']
    vectorstore = inputs['vectorstore']

    print(f"\nRunning RAG pipeline for query: {query}")
    relevant_docs = vectorstore.similarity_search_with_score(query, k=3)

    print("\nTop 3 most relevant chunks:")
    context = ""
    for i, (doc, score) in enumerate(relevant_docs, 1):
        print(f"{i}. Relevance Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:200]}...")
        print()
        context += doc.page_content + "\n\n"

    # Prepare the prompt
    full_prompt = PROMPT.format(context=context, question=query)
    print("\nFull Prompt sent to the model:")
    print(full_prompt)
    print("\n" + "="*50 + "\n")

    # Get the response
    response = qa_chain({"query": query})
    return response['result']

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

But, if the question is generic, then go ahead and answer the question, for example, what is an electric vehicle?
"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# Create runnables from functions using RunnableLambda
fetch_html_runnable = RunnableLambda(fetch_html)
process_website_runnable = RunnableLambda(process_website)
print_sample_embeddings_runnable = RunnableLambda(print_sample_embeddings)
rag_pipeline_runnable = RunnableLambda(rag_pipeline)

# Compose runnables into a chain
website_processing_chain = (
    fetch_html_runnable
    | process_website_runnable
)

# Wrap the process_website_runnable with retries and fallbacks
process_website_with_retry = process_website_runnable.with_retry(max_attempts=3)

def alternative_process_website(html_content):
    print("Using alternative method to process HTML content.")
    # Alternative processing logic here
    return process_website(html_content)  # For demonstration, we call the same function

alternative_process_website_runnable = RunnableLambda(alternative_process_website)

process_website_with_fallback = process_website_with_retry.with_fallbacks([
    alternative_process_website_runnable
])

# Update the chain to use the runnable with retries and fallbacks
website_processing_chain = (
    fetch_html_runnable
    | process_website_with_fallback
)

# Function to create vectorstore
def create_vectorstore(texts):
    print("Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    print_sample_embeddings(texts, embeddings)
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store created.")
    return vectorstore

vectorstore_runnable = RunnableLambda(create_vectorstore)

# Full pipeline including vectorstore creation
full_processing_chain = (
    website_processing_chain
    | vectorstore_runnable
)

# Main execution block
if __name__ == "__main__":
    print("Welcome to the Enhanced Web Scraping RAG Pipeline.")

    while True:
        url = input("Please enter the URL of the website you want to query (or 'quit' to exit): ")
        if url.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break

        try:
            print("Processing website content...")

            # Run the full processing chain
            texts = full_processing_chain.invoke(url)

            if not texts:
                print("No content found on the website. Please try a different URL.")
                continue

            # Create QA chain
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=texts.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT}
            )

            # Prepare the full RAG pipeline runnable
            full_rag_pipeline = rag_pipeline_runnable.bind(qa_chain=qa, vectorstore=texts)

            print("\nRAG Pipeline initialized. You can now enter your queries.")
            print("Enter 'new' to query a new website or 'quit' to exit the program.")

            while True:
                user_query = input("\nEnter your query: ")
                if user_query.lower() == 'quit':
                    print("Exiting the program. Goodbye!")
                    exit()
                elif user_query.lower() == 'new':
                    break

                # Run the RAG pipeline
                result = full_rag_pipeline.invoke({"query": user_query})
                print(f"RAG Response: {result}")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try a different URL or check your internet connection.")
