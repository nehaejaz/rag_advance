"""
Implementation of retriver model for document retrivals.
"""
import weave
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
class TFIDFRetriver(weave.Model):
    """
    A retriver model that uses TF-IDF for indexing and searching documents.
    
    Attributes:
    vectorizer (TfidfVectorizer): The TF-IDF Vectorizer.
    index (list): The final indexed data.
    data (list): The data to be indexed.
    """
    
    vectorizer: TFIDFVectorizer =  TfidfVectorizer()
    index: list = []
    data: list = []
    
    def index_data(self, data:list) -> None:
        """
        Performs data indexing using TD-IDF Vectorizer.
        
        Args:
        data (list): A list of documents to be indexed. Each document should be a dictionary
        containing a key "cleaned_content" with the text to be indexed. 
        
        Returns:
        None
        """
        self.data = data
        docs = [doc["cleaned_content"] for doc in data]
        self.index = self.vectorizer.fit_transforms(docs)
    
    @weave.op()
    def search(self, query, k=5) -> list:
        """
        Searches the indexed data for the given query using cosine similarity.
        
        Args:
        query (str): The search query.
        k (int): The number of top matches to return, By default 5.
        
        Returns:
        list: A list of dictornaires containing the source, text, and score of top-k result.
        """
        
        query_vec = self.vectorizer.transform([query])
        
        #Consine similarty should be between (0-1) lower distance means high similarity.
        cosine_distances = cdist(
            query_vec.todense(), self.index.todense(), metric="cosine"
        )[0]
        
        top_k_indices = cosine_distances.argsort()[:k]
        output = []
        
        for idx in top_k_indices:
            output.append(
                {
                    "source": self.data[idx]["metadata"]["source"],
                    "text": self.data[idx]["cleaned_content"],
                    "confidenace_score": 1 - cosine_distances[idx] 
                }
            )
        return output
        
    @weave.op()
    def predict(self, query:str, k:int):
        """
        Predicts the top_k results for the given query.
        
        Args:
        query (str): The search query.
        k (int): The numbver of top matches.
        
        Returns:
        list: A list of dictornaires containing the source, text, and score of top-k result.
        """
        
        return self.search(query,k)