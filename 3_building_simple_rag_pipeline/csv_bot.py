import os
import numpy as np
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration variables
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS = 150
MODEL_NAME = "gpt-4o-mini"  # You can change this to "gpt-4" if you have access
TEMPERATURE = 0.4

# Set up OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_csv_data(file_path):
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from CSV")
    return documents

def create_embeddings(documents):
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    embeddings = OpenAIEmbeddings()
    
    # Print sample embedding
    if texts:
        sample_text = texts[0].page_content
        sample_embedding = embeddings.embed_query(sample_text)
        print("\nSample Text:")
        print(sample_text)
        print("\nSample Embedding (first 10 dimensions):")
        print(np.array(sample_embedding[:10]))
        print(f"\nEmbedding shape: {np.array(sample_embedding).shape}")

    return texts, embeddings

def create_vectorstore(texts, embeddings):
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def process_query(query, qa_chain):
    result = qa_chain({"query": query})
    return result['result'], result['source_documents']

def main():
    print("Welcome to the CSV RAG Pipeline!")
    
    csv_path = input("Please enter the path to your CSV file: ")
    
    print("Loading and processing CSV data...")
    documents = load_csv_data(csv_path)
    
    print("Creating embeddings...")
    texts, embeddings = create_embeddings(documents)
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(texts, embeddings)
    
    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(vectorstore)
    
    print("\nRAG Pipeline initialized. You can now ask questions about your CSV data.")
    print("Enter 'quit' to exit the program.")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break
        
        answer, sources = process_query(query, qa_chain)
        print(f"\nAnswer: {answer}")
        print("\nSources:")
        for source in sources:
            print(f"- {source.page_content[:100]}...")

if __name__ == "__main__":
    main()