from langchain_openai import ChatOpenAI
from clarifai.client.model import Model
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate
)
from retriever import load_model, initialize_pinecone
import dotenv
from utils import get_content
import os


def init():
    dotenv.load_dotenv()
    global embed_model, pinecone_index
    embed_model =load_model()
    pinecone_index = initialize_pinecone(os.getenv('PINECONE_API_KEY'))
    
    # st.set_page_config(
    #     page_title="PyBot",
    #     page_icon="🐍",
    #     layout="centered",
    #     menu_items={
    #         "Get Help": "mailto:ferdinandchidera49@gmail.com",
    #         "Report a bug": "mailto:ferdinandchidera49@gmail.com",
    #     },
    # )
    
    
  
def main():
    
    # st.html("<h3 style='text-align: center;'>PyBot 🐍</h3>")
    
    model_url = "https://clarifai.com/openai/chat-completion/models/GPT-4"
    chat_model = ChatOpenAI(temperature=0.5)

    template = """
    You are a helpful Python-ONLY assistant that only writes Python Programming codes. You are not allowed to respond to questions or requests that are not related to Python programming.  You will:

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
    
    question = """
    Connect to a database server and print its version
    Note:

    Write SQL query to get the database server version.
    Connect to the database and use cursor.execute() to execute this query.
    Next, use cursor.fetchone() to fetch the record.

    """
    context = get_content(question, embed_model, pinecone_index)
    try:
        inference_params = dict(temperature=0.5, max_tokens=1024)
        model_output = Model(url=model_url,pat=os.getenv("CLARIFAI_API_KEY")).predict_by_bytes(prompt_template.format(context=context, question=question).encode(), input_type="text", inference_params=inference_params)
        response = model_output.outputs[0].data.text.raw
        return response
    except Exception as e1:
        try:
            review_chain = prompt_template | chat_model
            
            response = review_chain.invoke({"context":context, "question": question})
            return response.content
        except Exception as e2:
            return Exception("An error occurred while generating AI response, please contact the developer.")
                
if __name__ == "__main__":
    init()
    print(main())