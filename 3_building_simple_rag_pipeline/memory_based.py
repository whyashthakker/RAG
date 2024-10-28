import os
import bs4
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
WEBSITE_URL = "https://www.snapy.ai/"

# Initialize OpenAI components
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
embeddings = OpenAIEmbeddings()

def fetch_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None

def scrape_website(url):
    content = fetch_website_content(url)
    if not content:
        return []
    
    soup = bs4.BeautifulSoup(content, 'html.parser')
    text_content = soup.get_text(separator='\n', strip=True)
    return [text_content]

def process_website(url):
    docs = scrape_website(url)
    if not docs:
        print("No content could be extracted from the website.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_text(docs[0])
    
    if not splits:
        print("No valid text chunks were created.")
        return None
    
    vectorstore = FAISS.from_texts(splits, embedding=embeddings)
    return vectorstore

# Process the website
vectorstore = process_website(WEBSITE_URL)

if not vectorstore:
    print("Failed to create vectorstore. Exiting.")
    exit(1)

# Create a prompt template
prompt_template = """Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Human: {question}

Assistant: """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

# Chatbot function
def chatbot(query):
    response = qa_chain({"question": query})
    return response['answer']

# Example usage
if __name__ == "__main__":
    print("Welcome to the Website Chatbot!")
    print(f"This chatbot can answer questions about the content from: {WEBSITE_URL}")
    print("Type 'quit' to exit the chat.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Thank you for using the Website Chatbot. Goodbye!")
            break
        
        response = chatbot(user_input)
        print(f"\nChatbot: {response}")