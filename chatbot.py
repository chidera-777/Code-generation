from langchain_openai import ChatOpenAI
from clarifai.client.model import Model
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from retriever import load_model, initialize_pinecone
import streamlit as st
import dotenv
from utils import get_content, is_new_question, format_messages, handle_message_updates, display_new_message
import asyncio
import json
import os


def init():
    dotenv.load_dotenv()
    global embed_model, pinecone_index
    embed_model =load_model()
    pinecone_index = initialize_pinecone(os.getenv('PINECONE_API_KEY'))
    
    st.set_page_config(
        page_title="PyBot",
        page_icon="🐍",
        layout="centered",
        menu_items={
            "Get Help": "mailto:ferdinandchidera49@gmail.com",
            "Report a bug": "mailto:ferdinandchidera49@gmail.com",
        },
    )
    
    
  
def main():
    
    st.html("<h3 style='text-align: center;'>PyBot 🐍</h3>")
    
    model_url = "https://clarifai.com/openai/chat-completion/models/GPT-4"
    chat_model = ChatOpenAI(temperature=0.5)

    prompt = """
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
    - Use the provided context: {context} and your base knowledge in answering ONLY PYTHON-related questions

    5. If you're unsure about a Python-related question, respond with exactly: "Sorry, I cannot provide a reliable answer to this Python question."
    6. Do not go outside the context of the user's query
    """
    
    if "all_messages" not in st.session_state:
        st.session_state.all_messages = [
            AIMessage(content="Hi there! I'm PyBot, your helpful Python assistant. How can I assist you today?"),
            SystemMessage(content=prompt),
        ]
        
        st.session_state.current_context_messages = [
            SystemMessage(content=prompt),
        ]
        
        st.session_state.previous_question = ""
        
    with st.chat_message("ai", avatar="bot_img.jpg"):
        st.write(f"<p style='font-size:14px; font-weight:600;'>{st.session_state.all_messages[0].content}</p>", unsafe_allow_html=True)
       
    chat_container = st.container()
    question = st.chat_input("Ask me anything about Python programming")
    with chat_container:
        asyncio.run(handle_message_updates(st.session_state.all_messages))
            
    if question:
        reset_context = is_new_question(embed_model, question, st.session_state.previous_question)
        
        if reset_context:
            
            #Checks if the new question has similar context with the previous question
            st.session_state.current_context_messages = [
                SystemMessage(content=prompt),
            ]
        
        st.session_state.current_context_messages.append(HumanMessage(content=question))
        st.session_state.all_messages.append(HumanMessage(content=question))
        
        #Automatically update display when user input is added
        with chat_container:
            asyncio.run(display_new_message(st.session_state.all_messages[-1], len(st.session_state.all_messages)-1))
        
        #Formats the current context messages session state to be sent to te model
        message_hist = format_messages(st.session_state.current_context_messages)
        messages_str = json.dumps(message_hist)
                    
        with st.spinner("Generating response..."):          
            context = get_content(question, embed_model, pinecone_index)
            st.session_state.current_context_messages[0].content = f"{prompt}\n\n Context: \n{context}"
            
            # Query the MODEL
            try:
                inference_params = dict(temperature=0.5, max_tokens=1024)
                model_output = Model(url=model_url,pat=os.getenv("CLARIFAI_API_KEY")).predict_by_bytes(messages_str.encode(), input_type="text", inference_params=inference_params)
                
                #Format the desired AI response 
                response = model_output.outputs[0].data.text.raw
                clean_res = response.replace("Assistant: ", "")
            except Exception as e1:
                try:
                    res = chat_model(messages_str)
                    clean_res = res.content
                    print(clean_res)
                except Exception as e2:
                    st.error("An error occurred while generating AI response, please contact the developer.", icon="⚠️")
                    return
            
            st.session_state.current_context_messages.append(AIMessage(content=clean_res))
            st.session_state.all_messages.append(AIMessage(content=clean_res))
            st.session_state.previous_question = question
            
            with chat_container:
                asyncio.run(display_new_message(st.session_state.all_messages[-1], len(st.session_state.all_messages)-1))
                
if __name__ == "__main__":
    init()
    main()