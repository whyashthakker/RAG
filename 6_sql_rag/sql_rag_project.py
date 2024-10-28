import os
import sqlite3
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain_core.messages import SystemMessage, HumanMessage

# Configuration
DB_PATH = "6_sql_rag/tesla_motors_data.db"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def format_sql_results(results: List[tuple], columns: List[str]) -> str:
    """Format SQL query results into a readable string"""
    if not results:
        return "No results found"
    
    output = []
    for row in results:
        row_dict = dict(zip(columns, row))
        formatted_row = "\n".join([f"{col}: {val}" for col, val in row_dict.items()])
        output.append(formatted_row)
    
    return "\n\n".join(output)

def create_tesla_agent():
    """Create a SQL agent for Tesla database"""
    
    # Initialize the database connection
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    
    # Initialize the language model
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        max_tokens=1000
    )
    
    # Create the SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Custom system message for Tesla-specific queries
    system_message = SystemMessage(content="""You are an AI assistant specialized in querying Tesla Motors database. 
    The database contains information about Tesla vehicles including models, variants, prices, specifications, and sales data.
    
    When analyzing the data:
    1. Always format prices as currency with commas
    2. Express battery capacity in kWh and charging time in hours
    3. For any analysis, mention the sample size you're using
    4. Be specific about which models and variants you're discussing
    
    Guidelines for your responses:
    1. If asked about trends, include specific numbers and percentages
    2. For price analysis, mention both average and range
    3. When comparing models, cite specific differences in specifications
    4. If the data is insufficient, clearly state what's missing
    
    Before executing any query:
    1. Verify table and column names
    2. Check query syntax
    3. Ensure proper joins if needed
    4. Add appropriate limiting clauses
    
    DO NOT make any modifications to the database (no INSERT, UPDATE, DELETE, etc.).""")

    # Create the agent
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="openai-tools",
        verbose=True,
        system_message=system_message,
    )
    
    return agent

def main():
    print("Welcome to the Tesla Motors Database Agent")
    print("Loading agent...")
    
    try:
        agent = create_tesla_agent()
        
        print("\nAgent initialized successfully!")
        print("\nYou can ask questions about Tesla vehicles, such as:")
        print("- What's the average price of Tesla Model S?")
        print("- How many vehicles were sold in California?")
        print("- Which model has the highest battery capacity?")
        print("- Compare the specifications of Model S and Model X")
        print("\nType 'quit' to exit")
        
        while True:
            question = input("\nWhat would you like to know about Tesla vehicles? ").strip()
            
            if question.lower() == 'quit':
                print("Thank you for using the Tesla Motors Database Agent. Goodbye!")
                break
                
            try:
                # Get response from agent
                response = agent.invoke({"input": question})
                
                # Print the response
                print("\nAnswer:", response["output"])
                
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                print("Please try rephrasing your question.")
                
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        print("Please check your database connection and API key.")

if __name__ == "__main__":
    main()