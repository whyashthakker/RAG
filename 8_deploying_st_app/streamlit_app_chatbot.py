#pip install streamlit requests langchain-community lxml beautifulsoup4 langchain langchain-openai faiss-cpu numpy openai

import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import numpy as np
import tempfile
from langchain_community.document_loaders import BSHTMLLoader
import streamlit as st

# Configuration variables
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_TOKENS = 8192
MODEL_NAME = "gpt-4o-mini"  # Corrected model name
TEMPERATURE = 0.4

# Set up OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_api_key():
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    else:
        return st.sidebar.text_input("Enter your OpenAI API key:", type="password")

def scrape_website(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/91.0.4472.124 Safari/537.36'
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
            st.warning("No content found. The website might have an unusual structure or require JavaScript.")
            return []

        return content
    except requests.RequestException as e:
        st.error(f"Error scraping the website: {e}")
        return []

def clean_content(content_list):
    # Remove very short or common unwanted items
    cleaned = [text for text in content_list if len(text) > 20 and not any(
        item in text.lower() for item in ['sign up', 'sign in', 'cookie', 'privacy policy'])]
    return cleaned

def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.error(f"Error fetching the website: {e}")
        return None

def process_website(url):
    html_content = fetch_html(url)
    if not html_content:
        st.error("No content could be fetched from the website.")
        return []

    # Use a temporary file to store the HTML content
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    try:
        # Try to use BSHTMLLoader with default settings (which uses 'lxml')
        loader = BSHTMLLoader(temp_file_path)
        documents = loader.load()
    except ImportError:
        st.warning("'lxml' is not installed. Falling back to built-in 'html.parser'.")
        # If 'lxml' is not available, use the built-in 'html.parser'
        loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
        documents = loader.load()

    # Clean up the temporary file
    os.unlink(temp_file_path)

    st.write(f"\n**Number of documents loaded:** {len(documents)}")
    if documents:
        st.write("**Sample of loaded content:**")
        st.text(documents[0].page_content[:200] + "...")
        st.write(f"**Metadata:** {documents[0].metadata}")

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    st.write(f"**Number of text chunks after splitting:** {len(texts)}")
    return texts

def print_sample_embeddings(texts, embeddings):
    if texts:
        sample_text = texts[0].page_content
        sample_embedding = embeddings.embed_query(sample_text)
        st.write("\n**Sample Text:**")
        st.text(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)
        st.write("\n**Sample Embedding (first 10 dimensions):**")
        st.write(np.array(sample_embedding[:10]))
        st.write(f"\n**Embedding shape:** {np.array(sample_embedding).shape}")
    else:
        st.warning("No texts available for embedding sample.")

def rag_pipeline(query, qa_chain, vectorstore):
    relevant_docs = vectorstore.similarity_search_with_score(query, k=5)

    st.write("\n**Top 3 most relevant chunks:**")
    context = ""
    for i, (doc, score) in enumerate(relevant_docs, 1):
        st.write(f"**{i}. Relevance Score:** {score:.4f}")
        st.text(f"   Content: {doc.page_content[:200]}...")
        st.write("")
        context += doc.page_content + "\n\n"

    # Create full prompt (optional: for debugging)
    # full_prompt = PROMPT.format(context=context, question=query)
    # st.write("\n**Full Prompt sent to the model:**")
    # st.text(full_prompt)
    # st.write("\n" + "="*50 + "\n")

    response = qa_chain.invoke({"query": query})
    return response['result']

# Set up the OpenAI language model and prompt template
def setup_qa_chain(embeddings, llm):
    template = """Context: {context}

Question: {question}

Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

But, if the question is generic, then go ahead and answer the question, example what is an electric vehicle?
"""
    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    vectorstore = FAISS.from_documents([], embeddings)  # Initialize empty vectorstore
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa, vectorstore, PROMPT

def main():
    st.set_page_config(page_title="Web Scraping RAG Pipeline", layout="wide")
    st.title("📄 Enhanced Web Scraping RAG Pipeline")
    st.write("Retrieve information from any website and ask questions based on its content.")

    # Sidebar for API Key
    st.sidebar.header("Configuration")
    api_key = get_api_key()
    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API key to proceed.")
        st.stop()
    else:
        os.environ["OPENAI_API_KEY"] = api_key

    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    if 'PROMPT' not in st.session_state:
        # Define the prompt template
        template = """Context: {context}

Question: {question}

Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

But, if the question is generic, then go ahead and answer the question, example what is an electric vehicle?
"""
        st.session_state.PROMPT = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    # Input for URL
    url = st.text_input("Enter the URL of the website you want to query:", "")

    if st.button("Process Website"):
        if not url:
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Processing website content..."):
                texts = process_website(url)
                if texts:
                    with st.spinner("Creating embeddings and vector store..."):
                        embeddings = st.session_state.embeddings
                        st.session_state.vectorstore = FAISS.from_documents(texts, embeddings)
                        # Update the QA chain with the new vectorstore
                        st.session_state.qa_chain = RetrievalQA.from_chain_type(
                            llm=st.session_state.llm,
                            chain_type="stuff",
                            retriever=st.session_state.vectorstore.as_retriever(),
                            chain_type_kwargs={"prompt": st.session_state.PROMPT}
                        )
                    st.success("Website processed successfully!")
                    st.info("You can now enter your queries below.")
                else:
                    st.error("No content found on the website. Please try a different URL.")

    # Query Section
    if st.session_state.qa_chain:
        st.markdown("---")
        st.header("🔍 Ask a Question")
        user_query = st.text_input("Enter your query:", "")

        if st.button("Get Answer"):
            if not user_query:
                st.error("Please enter a query.")
            else:
                with st.spinner("Fetching answer..."):
                    result = rag_pipeline(user_query, st.session_state.qa_chain, st.session_state.vectorstore)
                st.success("**RAG Response:**")
                st.write(result)

    # Option to reset the app
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset App"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    main()
