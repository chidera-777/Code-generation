from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import dotenv
import time
import os

def timer_decorator(func):
    """A simple decorator to time functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)  # Call the function
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"{func.__name__} took {elapsed_time:.2f} seconds")
        return result  # Return the result of the function
    return wrapper

dotenv.load_dotenv()

model_path = 'emebed-model'

@timer_decorator
def load_model():
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return model


@timer_decorator
def initialize_pinecone(api_key):
    pc = Pinecone(api_key=api_key)
    return pc.Index("autocomplete-code-index")


@timer_decorator
def search_docs(query, model, index, k=3):
    
    qs = model.embed_query(query)
    
    results = index.query(vector = qs, top_k = k, include_metadata = True)
    return results["matches"] if results["matches"][0]["score"] >= 0.5 else []
