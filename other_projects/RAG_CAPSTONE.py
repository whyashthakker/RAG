import streamlit as st
from sentence_transformers import SentenceTransformer
import openai
import re
import pdfplumber
from pinecone import Pinecone, ServerlessSpec
import os

from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index exists and create it if not
index_name = 'ragfinalproj'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Adjust the dimension based on your embedding model
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)
# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define functions
def extract_text_from_pdf(pdf_file):
    full_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"  # Add newline to separate pages
    return full_text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.replace('...', '')  # Remove ellipsis
    return text

def split_into_chunks(text, chunk_size=2000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        if current_size + len(word) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for the space
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def store_embeddings(chunks):
    for i, chunk in enumerate(chunks):
        embedding = sentence_model.encode(chunk).tolist()
        index.upsert(vectors=[(f"chunk_{i}", embedding, {"text": chunk})])

def retrieve_relevant_chunks(query, k=3):
    query_embedding = sentence_model.encode([query]).tolist()
    results = index.query(vector=query_embedding[0], top_k=k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

def generate_response(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

def rag_model(query):
    relevant_chunks = retrieve_relevant_chunks(query, k=2)
    context = " ".join(relevant_chunks)[:2000]
    response = generate_response(query, context)
    return response

# Streamlit interface
st.title("Car Manual Q&A Assistant")
st.write("Upload your car manual PDF and ask questions about it.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.write("Processing your PDF...")
    
    # Extract and clean text from PDF
    full_text = extract_text_from_pdf(uploaded_file)
    full_text = clean_text(full_text)
    
    # Split text into chunks and store embeddings
    chunks = split_into_chunks(full_text)
    store_embeddings(chunks)
    
    st.write("PDF processed and embeddings stored. You can now ask questions about the manual.")

    query = st.text_input("Enter your question here:")

    if query:
        response = rag_model(query)
        st.write("Answer:", response)
