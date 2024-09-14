import streamlit as st
from sentence_transformers import SentenceTransformer
import pinecone
import openai
import re
import pdfplumber
from pinecone import Pinecone, ServerlessSpec
import os

# Set up OpenAI
openai.api_key = "your_api_key"

# Initialize Pinecone

pc = Pinecone(api_key="your_apikey")
index = pinecone.Index(index_name="ragcapstone", host="your host",api_key="your_apikey")

# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, k=3):
    query_embedding = sentence_model.encode([query]).tolist()
    results = index.query(vector=query_embedding[0], top_k=k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

# Function to generate a response using OpenAI
def generate_response(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

# Main function for RAG model
def rag_model(query):
    relevant_chunks = retrieve_relevant_chunks(query, k=2)
    context = " ".join(relevant_chunks)[:2000]
    response = generate_response(query, context)
    return response

# Streamlit interface
st.title("Car Manual Q&A Assistant")
st.write("Ask questions about your car manual, and I'll provide answers based on the information available.")

query = st.text_input("Enter your question here:")

if query:
    response = rag_model(query)
    st.write("Answer:", response)
