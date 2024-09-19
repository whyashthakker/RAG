import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import tempfile
from langchain.document_loaders import BSHTMLLoader

# Configuration variables
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS = 15000
MODEL_NAME = "gpt-3.5-turbo"  # Changed from "gpt-4o-mini" to a standard OpenAI model
TEMPERATURE = 0.4

# Set up OpenAI API key
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def fetch_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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
        raise ValueError("No content could be fetched from the website.")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    try:
        loader = BSHTMLLoader(temp_file_path)
        documents = loader.load()
    except ImportError:
        st.warning("'lxml' is not installed. Falling back to built-in 'html.parser'.")
        loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
        documents = loader.load()

    os.unlink(temp_file_path)

    st.write(f"Number of documents loaded: {len(documents)}")
    if documents:
        st.write("Sample of loaded content:")
        st.write(documents[0].page_content[:200] + "...")
    
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    st.write(f"Number of text chunks after splitting: {len(texts)}")
    return texts

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
    
    st.write("Top 3 most relevant chunks:")
    context = ""
    for i, (doc, score) in enumerate(relevant_docs, 1):
        st.write(f"{i}. Relevance Score: {score:.4f}")
        st.write(f"   Content: {doc.page_content[:200]}...")
        context += doc.page_content + "\n\n"

    response = qa_chain({"query": query})
    return response['result']

def main():
    st.title("Web Content RAG Chatbot")

    # API Key input
    if not st.session_state.OPENAI_API_KEY:
        st.session_state.OPENAI_API_KEY = st.text_input("Enter your OpenAI API key:", type="password")
        if st.session_state.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY

    if not st.session_state.OPENAI_API_KEY:
        st.warning("Please enter your OpenAI API key to continue.")
        return

    # URL input
    url = st.text_input("Enter the URL of the website you want to query:")

    if url:
        if 'vectorstore' not in st.session_state or st.session_state.last_url != url:
            with st.spinner("Processing website content..."):
                try:
                    texts = process_website(url)
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_documents(texts, embeddings)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.last_url = url
                    st.success("Website processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    return

        # Set up the language model and QA chain
        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Chat interface
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What would you like to know about the website?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_pipeline(prompt, qa, st.session_state.vectorstore)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()