from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import re, time
from retriever import search_docs
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from streamlit_chat import message



def get_content(question, model, index):
    matches =  search_docs(question, model, index)
    contexts = []
    if matches:
        for match in matches:
            if "metadata" in match and "text" in match["metadata"]:
                contexts.append(match['metadata']['text'])
            else:
                print(f"Unexpected match format {match}")
    else:
        print("No matches found")

    context = "".join(contexts)
    return context


def is_new_question(model, new_question, old_question, threshold=0.5):
    """"
        Checks if the new question is similar to the previous question, if no previous question is provided, returns True(A new question)
    Args:
        model (object): The model to use for embedding.
        new_question (str): The new question to check.
        old_question (str): The previous question.
        threshold (float): The threshold for cosine similarity.
    Returns:
        bool: True if the new question is not similar to the previous question, False otherwise.
    """
    if not old_question:
        return True
    
    prev_question_embedding = model.encode(old_question)
    new_question_embedding = model.encode(new_question)
    cos_sim = cosine_similarity([prev_question_embedding], [new_question_embedding])[0][0]
 
    return cos_sim < threshold


def format_messages(messages):
    """"
        Formats the Context message to be sent to the model
    Args:
        messages (list): The list of messages to be formatted.
    Returns:
        list: The formatted messages.
    
    """
    formatted_messages = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            formatted_messages.append({
                "role": "system",
                "content": msg.content
            })
        elif isinstance(msg, HumanMessage):
            formatted_messages.append({
                "role": "user",
                "content": msg.content
            })
        elif isinstance(msg, AIMessage):
            formatted_messages.append({
                "role": "assistant",
                "content": msg.content
            })
    
    return formatted_messages


def display_ai_message(content):
    """
        Displays the AI message codes in a code block.
    Args:
        content (str): The content of the AI message.
    Returns:
        the AI message
    """
    parts = re.split(r'(```python[\s\S]*?```)', content)
    for part in parts:
        if part.startswith('```python') and part.endswith('```'):
            code = part[9:-3].strip()
            st.code(code, language='python')
        else:
            escaped_part = re.sub(r'^\s*#', '', part, flags=re.MULTILINE)
            st.write(escaped_part)
             
async def handle_message_updates(messages):
    """
    Asynchronously handles message updates and displays them in the chat interface
    Uses timestamp-based unique keys to prevent duplicate key errors
    """
    message_placeholder = st.empty()
    
    # Get current timestamp to create unique keys
    timestamp = int(time.time() * 1000)
    
    async def update_messages():
        for i, msg in enumerate(messages[2:]):  # Skip first 2 system messages
            # Create unique key using timestamp and message index
            unique_key = f"msg_{timestamp}_{i}"
            
            if i % 2 == 0:
                message(
                    msg.content, 
                    is_user=True, 
                    avatar_style="avataaars", 
                    key=unique_key
                )
            else:
                with st.chat_message("ai", avatar="bot_img.jpg"):
                    display_ai_message(msg.content)
            # Small delay to create smooth animation effect
            await asyncio.sleep(0.1)

    await update_messages()
               
async def display_new_message(msg, index):
    """
    Display a single new message with a unique key
    """
    timestamp = int(time.time() * 1000)
    unique_key = f"msg_{timestamp}_{index}"
    
    if isinstance(msg, HumanMessage):
        message(
            msg.content, 
            is_user=True, 
            avatar_style="avataaars", 
            key=unique_key
        )
    elif isinstance(msg, AIMessage):
        with st.chat_message("ai", avatar="bot_img.jpg"):
            display_ai_message(msg.content)
    
    await asyncio.sleep(0.1)
