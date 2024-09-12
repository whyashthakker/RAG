import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import streamlit as st

# Web scraping function
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return [p.get_text() for p in paragraphs]

# Clean scraped content
def clean_content(content_list):
    unwanted_items = {'Sign up', 'Sign in', 'Follow', '--', '15', 'Listen'}
    cleaned = [text for text in content_list if text and text not in unwanted_items]
    return cleaned

# Convert cleaned content to Documents
def convert_to_documents(content_list):
    return [Document(page_content=text) for text in content_list]

# Create embeddings and vector store
def create_vectorstore(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

# Set up the language model
def setup_llm():
    model_name = "gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True
    )
    return HuggingFacePipeline(pipeline=pipe)

# Set up the QA system
def setup_qa(vectorstore, local_llm):
    template = """Context: {context}
    Question: {question}
    Answer the question concisely in one sentence based only on the given context:"""
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=local_llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

# Streamlit app
st.title("PDF Question Answering System")

url = st.text_input("Enter the URL to scrape:", "https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7")
question = st.text_input("Enter your question:")

if url and question:
    web_content = scrape_website(url)
    cleaned_content = clean_content(web_content)
    documents = convert_to_documents(cleaned_content)
    vectorstore = create_vectorstore(documents)
    local_llm = setup_llm()
    qa = setup_qa(vectorstore, local_llm)

    response = qa.run(question)
    st.write(f"Answer: {response}")
