import os
import sqlite3
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import SQLiteVectorStore
from langchain.docstore.document import Document
import numpy as np
import tempfile
from langchain_community.document_loaders import BSHTMLLoader

# Configuration variables
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS = 15000
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.4
DB_PATH = "website_content.db"

# Set up OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS websites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        content TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        website_id INTEGER,
        chunk_text TEXT,
        embedding BLOB,
        FOREIGN KEY (website_id) REFERENCES websites (id)
    )
    ''')
    conn.commit()
    conn.close()

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

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    return texts

def store_website_content(url, texts):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Store website content
    cursor.execute("INSERT OR REPLACE INTO websites (url, content) VALUES (?, ?)",
                   (url, "\n".join([text.page_content for text in texts])))
    website_id = cursor.lastrowid
    
    # Generate and store embeddings
    embeddings = OpenAIEmbeddings()
    for text in texts:
        embedding = embeddings.embed_query(text.page_content)
        cursor.execute("INSERT INTO embeddings (website_id, chunk_text, embedding) VALUES (?, ?, ?)",
                       (website_id, text.page_content, np.array(embedding).tobytes()))
    
    conn.commit()
    conn.close()

def load_website_content(url):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, content FROM websites WHERE url = ?", (url,))
    result = cursor.fetchone()
    
    if result:
        website_id, content = result
        cursor.execute("SELECT chunk_text, embedding FROM embeddings WHERE website_id = ?", (website_id,))
        chunks = cursor.fetchall()
        
        texts = [Document(page_content=chunk[0], metadata={}) for chunk in chunks]
        embeddings = [np.frombuffer(chunk[1]) for chunk in chunks]
        
        conn.close()
        return texts, embeddings
    else:
        conn.close()
        return None, None

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
    template=template, input_variables=["context", "question"])

def rag_pipeline(query, qa_chain, vectorstore):
    relevant_docs = vectorstore.similarity_search_with_score(query, k=3)
    
    print("\nTop 3 most relevant chunks:")
    context = ""
    for i, (doc, score) in enumerate(relevant_docs, 1):
        print(f"{i}. Relevance Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:200]}...")
        print()
        context += doc.page_content + "\n\n"

    full_prompt = PROMPT.format(context=context, question=query)
    print("\nFull Prompt sent to the model:")
    print(full_prompt)
    print("\n" + "="*50 + "\n")

    response = qa_chain.invoke({"query": query})
    return response['result']

if __name__ == "__main__":
    print("Welcome to the SQL-based Web Scraping RAG Pipeline.")
    create_database()
    
    while True:
        url = input("Please enter the URL of the website you want to query (or 'quit' to exit): ")
        if url.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break
        
        try:
            texts, embeddings = load_website_content(url)
            
            if not texts:
                print("Processing website content...")
                texts = process_website(url)
                
                if not texts:
                    print("No content found on the website. Please try a different URL.")
                    continue
                
                print("Storing website content and creating embeddings...")
                store_website_content(url, texts)
                texts, embeddings = load_website_content(url)
            
            print("Creating vector store...")
            vectorstore = SQLiteVectorStore(DB_PATH)
            
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