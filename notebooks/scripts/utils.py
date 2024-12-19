"""
Utility function for RAG notebooks
"""
import inspect
from rich.syntax import Syntax
from rich.console import Console
import requests

TOKENIZERS={
    "command-r": "https://storage.googleapis.com/cohere-public/tokenizers/command-r.json",
    "command-r-plus": "https://storage.googleapis.com/cohere-public/tokenizers/command-r-plus.json",
}
def display_source(symbol):
    """
    Display the source code of a given symbol using rich's syntax highlighting
    
    Args: 
        symbol: The symbol (function, class, etc) whose souce code is to be highlighted
    
    Returns:
        None
    """
    
    try:
        source = inspect.getsource(symbol)
    except TypeError:
        print(
            f"Unable to get source code for {symbol}. It might be a built-in or compiled object."
        )
        return
    
    syntax = Syntax(source, "python", theme="monokai", line_numbers=True)
    console = Console()
    console.print(syntax)
    
def get_special_tokens_set(tokenizer_url=TOKENIZERS["command-r"]):
    """
    Fetches the special tokens set from the given Tokenizer url
    
    Args:
    tokenizer_url: The url to fetch the tokenier from
    
    Return:
    set: A set of special tokens
    """
    
    # https://docs.cohere.com/docs/tokens-and-tokenizers
    response = requests.get(tokenizer_url)
    return set([tok["content"] for tok in response.json()["added_tokens"]])
