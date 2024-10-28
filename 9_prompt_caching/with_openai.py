import os
import time
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain.docstore.document import Document
import numpy as np
import tempfile
from typing import Dict, List, Tuple
import hashlib
from langchain_community.document_loaders import BSHTMLLoader

# Configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MAX_TOKENS = 15000
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.4

# Set up cache
set_llm_cache(InMemoryCache())
query_cache: Dict[str, Tuple[str, float]] = {}

class RAGQueryProcessor:
    def __init__(self):
        # Set up OpenAI API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.api_key = input("Please enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize language model and embeddings
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize prompt template
        self.prompt = PromptTemplate(
            template="""Context: {context}

Question: {question}

Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

Your response should be detailed and specific, citing information from the context when possible.""",
            input_variables=["context", "question"]
        )

    def get_cache_key(self, query: str, context: str) -> str:
        """Generate a cache key from query and context"""
        combined = f"{query}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()

    def process_website(self, url: str) -> List[Document]:
        """Process website content and return documents"""
        print("\nProcessing website content...")
        start_time = time.time()
        
        try:
            # Fetch and process HTML
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as temp_file:
                temp_file.write(response.text)
                temp_file_path = temp_file.name

            loader = BSHTMLLoader(temp_file_path)
            documents = loader.load()
            os.unlink(temp_file_path)

            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(documents)
            
            process_time = time.time() - start_time
            print(f"Website processed in {process_time:.2f} seconds")
            print(f"Generated {len(texts)} text chunks")
            
            return texts
            
        except Exception as e:
            print(f"Error processing website: {str(e)}")
            return []

    def process_query(self, query: str, vectorstore) -> Tuple[str, float, float]:
        """Process a query using RAG and return result with timing"""
        start_time = time.time()
        
        # Get relevant documents
        retrieval_start = time.time()
        relevant_docs = vectorstore.similarity_search_with_score(query, k=3)
        retrieval_time = time.time() - retrieval_start
        
        # Build context
        context = "\n\n".join([doc.page_content for doc, score in relevant_docs])
        
        # Check cache
        cache_key = self.get_cache_key(query, context)
        if cache_key in query_cache:
            cached_result, original_time = query_cache[cache_key]
            current_time = time.time() - start_time
            return f"[CACHED] {cached_result}", current_time, original_time, retrieval_time
        
        # Process query through RAG pipeline
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        # Get model response
        llm_start = time.time()
        response = qa_chain.invoke({"query": query})
        llm_time = time.time() - llm_start
        
        # Calculate timing and cache result
        total_time = time.time() - start_time
        query_cache[cache_key] = (response['result'], total_time)
        
        return response['result'], total_time, None, retrieval_time

def main():
    print("Welcome to RAG Query Processor with Timing")
    print("This system will process website content and answer questions with timing information")
    print("Type 'quit' to exit\n")

    processor = RAGQueryProcessor()

    while True:
        url = input("\nEnter website URL (or 'quit' to exit): ").strip()
        
        if url.lower() == 'quit':
            print("\nExiting program. Goodbye!")
            break
        
        try:
            # Process website
            texts = processor.process_website(url)
            if not texts:
                print("No content found. Please try a different URL.")
                continue
            
            # Create vector store
            print("\nCreating vector store...")
            start_time = time.time()
            vectorstore = FAISS.from_documents(texts, processor.embeddings)
            print(f"Vector store created in {time.time() - start_time:.2f} seconds")
            
            # Query loop
            query_count = 0
            while True:
                query_count += 1
                query = input(f"\nQuery #{query_count}: ").strip()
                
                if not query:
                    print("Query cannot be empty. Please try again.")
                    query_count -= 1
                    continue
                
                if query.lower() == 'quit':
                    print("\nExiting program. Goodbye!")
                    exit()
                elif query.lower() == 'new':
                    break
                
                print(f"\nProcessing query #{query_count}...")
                
                try:
                    result, total_time, cached_time, retrieval_time = processor.process_query(query, vectorstore)
                    
                    print("\nResults:")
                    print("-" * 50)
                    print(f"Response: {result}")
                    print("-" * 50)
                    print(f"Document retrieval time: {retrieval_time:.2f} seconds")
                    print(f"Total processing time: {total_time:.2f} seconds")
                    
                    if cached_time is not None:
                        print(f"Original query took: {cached_time:.2f} seconds")
                        print(f"Speed improvement: {((cached_time - total_time) / cached_time * 100):.1f}%")
                    
                except Exception as e:
                    print(f"Error processing query: {str(e)}")
                    query_count -= 1
                
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again with a different URL.")

if __name__ == "__main__":
    main()