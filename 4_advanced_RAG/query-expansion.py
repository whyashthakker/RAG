import os
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class ParaphrasedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""
    
    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )

class QueryExpander:
    """A class to handle query expansion using LangChain and OpenAI."""
    
    def __init__(self, api_key: str = None):
        """Initialize the QueryExpander with necessary components."""
        # Set up OpenAI API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("No API key provided. Set OPENAI_API_KEY environment variable or pass key to constructor.")
        
        # Define the system prompt
        self.system_prompt = """You are an expert at expanding user questions into multiple variations. \
            Perform query expansion. If there are multiple common ways of phrasing a user question \
            or common synonyms for key words in the question, make sure to return multiple versions \
            of the query with the different phrasings.

            If there are acronyms or words you are not familiar with, do not try to rephrase them.

            Return at least 3 versions of the question that maintain the original intent."""
        
        # Set up the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}")
        ])
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
        
        # Bind tools and create analyzer
        self.llm_with_tools = self.llm.bind_tools([ParaphrasedQuery])
        self.query_analyzer = (
            self.prompt 
            | self.llm_with_tools 
            | PydanticToolsParser(tools=[ParaphrasedQuery])
        )
    
    def expand_query(self, question: str) -> List[str]:
        """
        Expand a question into multiple paraphrased variations.
        
        Args:
            question (str): The original question to expand
            
        Returns:
            List[str]: List of paraphrased variations of the question
        """
        try:
            # Get paraphrased queries
            results = self.query_analyzer.invoke({"question": question})
            
            # Extract just the query strings
            variations = [result.paraphrased_query for result in results]
            
            return variations
            
        except Exception as e:
            print(f"Error expanding query: {str(e)}")
            return []

def main():
    """Example usage of the QueryExpander"""
    try:
        # Initialize the expander
        expander = QueryExpander()
        
        print("Welcome to LangChain Query Expander!")
        print("Enter a question to see different variations (or 'quit' to exit)")
        print("\nExample questions:")
        print("- How to use multi-modal models in a chain?")
        print("- What's the best way to stream events from an LLM agent?")
        print("- How to implement RAG with vector databases?")
        
        while True:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() == 'quit':
                print("Thank you for using LangChain Query Expander. Goodbye!")
                break
                
            print("\nGenerating variations...")
            variations = expander.expand_query(question)
            
            print("\nExpanded Queries:")
            for i, variation in enumerate(variations, 1):
                print(f"\n{i}. {variation}")
            
            print("\nTotal variations generated:", len(variations))
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main()