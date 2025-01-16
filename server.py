from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from clarifai.client.model import Model
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate
)
from retriever import load_model, initialize_pinecone
import dotenv
import asyncio
from utils import get_content
import dotenv
import os
import gc
import torch


# Global variables
embed_model = None
pinecone_index = None
model_ready = asyncio.Event()



class ModelManager():
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.embed_model = None
            self.pinecone_index = None
            ModelManager._initialized = True
            
    async def initialize(self):
        if self.embed_model is None:
            try:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                dotenv.load_dotenv()
                
                self.embed_model = load_model()
                self.pinecone_index = initialize_pinecone(os.getenv('PINECONE_API_KEY'))
                return True
            
            except Exception as e:
                print(f"Error initializing model: {e}")
                return False
        return True
    
    def get_models(self):
        return self.embed_model, self.pinecone_index
    
    def is_initialized(self):
        return self.embed_model is not None and self.pinecone_index is not None
    
    def cleanup(self):
        """Free up memory when needed"""
        if self.embed_model:
            del self.embed_model
            self.embed_model = None
        if self.pinecone_index:
            del self.pinecone_index
            self.pinecone_index = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    
model_manager = ModelManager()

class GenerateRequest(BaseModel):
    question: str
        

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the application.
    :param app: The FastAPI application.
    """
    print("Starting up...")
    gc.collect()
    
    try:
        await model_manager.initialize()
        model_ready.set()
    except Exception as e:
        print(f"Startup error: {str(e)}")
    
    yield
    
    print("Shutting down...")
    model_manager.cleanup()

app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generates a response based on the given question.
    :param question: The question to generate a response for.
    :return: The generated response.
    """
    question = request.question
    try:
        
        try:
            await asyncio.wait_for(model_ready.wait(), timeout=30)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503, 
                detail="Model is still initializing. Please try again later."
            )
        
        # Get initialized models
        embed_model, pinecone_index = model_manager.get_models()
        if not embed_model or not pinecone_index:
            raise HTTPException(
                status_code=503,
                detail="Models not properly initialized"
            )
            
        model_url = "https://clarifai.com/openai/chat-completion/models/GPT-4"
        chat_model = ChatOpenAI(temperature=0.5)

        template = """
        You are PyBot, a helpful Python-ONLY assistant that only writes Python Programming codes. You are not allowed to respond to questions or requests that are not related to Python programming.  You will:

        1. ONLY respond to questions about Python programming, code debugging, and Python-related tasks
        2. Avoid any responses that are not directly related to Python Programming.
        3. For ANY non-Python question, output ONLY this exact line, with no additional text, code, or any suggestion what so ever, just go straight to the point:
        "I am a Python-only assistant. I cannot help with non-Python questions. Please ask me something about Python programming."
        4. When answering Python questions:
        - Provide a clear, clean and readable Python code solution
        - Use proper code formatting with Markdown
        - Use proper indentation and take note of case sensitivity when naming variables, functions, and classes
        - Ensure code is accurate and tested
        - Go straight to the point and avoid outputting unecessary contents
        - Use the provided context and your base knowledge in answering ONLY PYTHON-related questions
        Here is the context:
        
        <context>
        {context}
        </context>

        5. If you're unsure about a Python-related question, respond with exactly: "Sorry, I cannot provide a reliable answer to this Python question."
        6. Do not go outside the context of the user's query
        7. If asked to identify yourself, respond with: "I am PyBot, a helpful Python-ONLY assistant that only writes Python Programming codes."
        """
        system_message_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context"],
                template=template
            )
        )
        
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["question"],
                template="{question}"
            )
        )
        messages = [
            system_message_prompt,
            human_message_prompt
        ]
        prompt_template = ChatPromptTemplate(
            input_variables=["context", "question"],
            messages=messages
        )
        
        context = get_content(question, embed_model, pinecone_index)     
        try:
            inference_params = dict(temperature=0.5, max_tokens=1024)
            model_output = Model(
                url=model_url,pat=os.getenv("CLARIFAI_API_KEY")).predict_by_bytes(prompt_template.format(context=context, question=question).encode(),
                input_type="text", 
                inference_params=inference_params
            )
            response = model_output.outputs[0].data.text.raw
            return {"response": response}
        except Exception as e1:
            try:
                review_chain = prompt_template | chat_model
                
                response = review_chain.invoke({"context":context, "question": question})
                return {"response": response.content}
            except Exception as e2:
                return HTTPException(f"An error occurred while generating AI response, please contact the developer. \nError:{e2}")
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


