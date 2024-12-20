from utils import get_special_tokens_set

def make_text_tokenization_safe(
    content:str, special_tokens_set:set = get_special_tokens_set()
)-> str: 
    """
    We are removing special tokens such as <START> <END> <SEP> from user content
    so that it shouldn't disrupt the model's interpretations. 
    
    Args: 
    content: String that is to be processed
    special_tokens_set: A set of special tokens that need to be removed 
    
    Return:
    str: A string with special tokens removed
    """
    print("TS",special_tokens_set)
    def remove_special_tokens(text: str)->str:
        """
        Removes special tokens from a given text
        
        Args:
        text: Text with special tokens
        
        Return:
        str: String with special tokens removed
        """
        
        for token in special_tokens_set:
            text = text.replace(token,"")
            # print(text)
        return text
    
    cleaned_content = remove_special_tokens(content)
    return cleaned_content
    
cleaned_content = make_text_tokenization_safe(content="I <|USER_5_TOKEN|> will see you at the end of today")
print("cleaned_content",cleaned_content)