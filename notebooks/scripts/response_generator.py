"""
A module containing resposnse generator using Cohere's Command-R model.
"""
import cohere
import os
import weave 
from typing import List, Dict

class SimpleResponseGenerator(weave.Model):
    """
    A simple response generator using Cohere's command-r API.
    
    Attribute:
    model (str): The model name to be used for generating the response.
    prompt (str): The prompt to be used for generating the response
    clinet (cohere.ClientV2): The cohere client for interacting with choere API.
    """
    
    model: str
    prompt: str
    client: cohere.ClientV2 = None
    
    def __init__(self, **kwargs):
        """
        Initializes the SimpleResponseGenerator with the provided key arguments.
        Sets up Cohere's Client using API ket from environment variables.
        """
        
        super().__init__(**kwargs)
        self.client = cohere.ClientV2(
            api_key=os.environ["COHERE_API_KEY"]
        )
    
    @weave.op()
    def generate_context(self, context: List[Dict[str,any]]) -> List[Dict[str,any]]:
        """
        Generates a list of context from the provided context list.
        
        Args:
        context (List[Dict[str,any]]): A list of dictonaries containing context data
        
        Returns:
        List[Dict[str,any]]: A list of dictonaries containing source and key 
        """
        
        context = [
            {"data": {"source": item["source"], "text":item["text"]}}
            for item in context
        ]
        
        return context
    
    @weave.op()
    def create_messages(self,query:str):
        """
        Create a list of chat messages for the chat model based on the query.
        
        Args:
        query (str): The user's query
        
        Returns:
        List[Dict[str,any]]: A list of messages formatted for the chat model
        """
        
        messages= [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": query},
        ]
        
        return messages
    
    @weave.op()
    def generate_response(self, query: str, context: List[Dict[str,any]]):
        """
        Generate a response for the chat model based on the user query and reterived context.
        
        Args:
        query (str): The user's query.
        context (List[Dict[str,any]]): A list of dictonaries containing context data
        
        Returns:
        str: The generated response from the chat model
        """
        documents = self.generate_context(context)
        messages = self.create_messages(query)
        response = self.client.chat(
            messages = messages,
            model = self.model ,
            temperature=0.1,
            max_tokens = 200,
            documents = documents,
        )
        return response.message.content[0].text
    
    @weave.op()
    def predict(self, query:str, context: List[Dict[str,any]]) -> str:
        """
        Predicts the response for the given query and prompt
        
        Args:
        quert (str): The user's query
        context: A list of disctionaries containing context data
        
        Returns:
        str: The predicted response from the chat model
        """
        return self.generate_response(query, context)
        