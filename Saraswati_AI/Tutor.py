# 




import streamlit as st
from fpdf import FPDF
import os
import json

from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph
import json
from langchain_core.messages import ToolMessage


from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langchain_community.tools import YouTubeSearchTool
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.messages import (
AIMessage,
BaseMessage,
ChatMessage,
FunctionMessage,
HumanMessage,
)
import json 
import requests
import functools
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import (
ChatPromptTemplate,
MessagesPlaceholder,
SystemMessagePromptTemplate,
HumanMessagePromptTemplate,
)
from streamlit_lottie import st_lottie
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper

# llm = ChatOpenAI(api_key="AIzaSyBdEYoKwmQr3rTkB_rD77D_QBYxfDjPJwQ",
#                  model="google/gemma-2b-it",temperature=0.05)
from langchain_google_vertexai import VertexAI
import vertexai
vertexai.init(project="saraswati-ai", location="us-central1")
  
llm = VertexAI(model_name="gemini-pro")
# message = "What are some of the pros and cons of Python as a programming language?"

st.title("Galileo AI 🤖")

if 'depth' not in st.session_state:
    st.session_state['depth'] = 'Elementary (Grade 1-6)'

if 'Learning' not in st.session_state:
    st.session_state['Learning'] = 'Verbal'

if 'Tone' not in st.session_state:
    st.session_state['Tone'] = 'Encouraging)'
if 'Reasoning' not in st.session_state:
    st.session_state['Reasoning'] = 'Deductive'

# Using object notation

st.session_state['depth'] = st.sidebar.selectbox(
    "Depth",
    ("Elementary (Grade 1-6)", "Middle School (Grade 7-9)", "Highschool (10-12)","College Prep","Undergraduate","Graduate")
)
st.session_state['Learning'] = st.sidebar.selectbox(
    "Learning Styles",
    ("Verbal", "Active", "Intuitive)","Reflective","Global")
)

st.session_state['Tone'] = st.sidebar.selectbox(
    "Tone Styles",
    ("Encouraging", "Neutral", "Informative","Friendly","Humorous")
)
st.session_state['Reasoning'] = st.sidebar.selectbox(
    "Reasoning Frameworks",
    ("Deductive", "Inductive", "Abductive","Analogical","Causal")
)






# Notice that "chat_history" is present in the prompt template
template1 = """
.
You are an AI Tutor assistant. your name is Galileo. Strict rule is that always don't greet user . just focus on conversation and content . Your goal is to have a friendly, helpful, and engaging conversation with the human to help them learn about the topic they want to study.

Then, based on the human's input about what they want to learn, you will provide a comprehensive and conversational tutorial, guiding them through the topic step-by-step.

Your responses should be tailored to the human's profile and learning style, and you should aim to make the interaction feel natural and interactive, as if you were a knowledgeable tutor teaching a student.

Please generate a response that demonstrates this conversational tutoring approach. YOu also move forward based on User personalization learning - 

Depth - {depth} , Learning Styles - {learning} , Tone Styles - {tone} , Reasoning Framework - {reasoning}
Previous conversation:
{chat_history}


New human question: {question}
Response:"""




prompt = PromptTemplate(
    template=template1,input_variables=['question','chat_history','depth','learning','tone','reasoning']
)
chain = prompt | llm



# Configuration	Options

# Communication	Format, Textbook, Layman, Story Telling, Socratic
# Tone Styles	Encouraging, Neutral, Informative, Friendly, Humorous
# Reasoning Frameworks	Deductive, Inductive, Abductive, Analogical, Causal
# Language	English (Default), any language GPT-4 is capable of doing.
    

@st.cache_data
def conversation_chat1(query):
    result = chain.invoke({"question": query,'chat_history':st.session_state['history1'],'depth':st.session_state['depth'],'learning':st.session_state['Learning'],'tone':st.session_state['Tone'],'reasoning':st.session_state['Reasoning']})
    st.session_state['history1'].append((st.session_state['past1'][-1], st.session_state['generated1'][-1]))
    
    
    return result

def initialize_session_state1():
    

    if 'generated1' not in st.session_state:
        st.session_state['generated1'] = ["Hello! Ask me anything about 🤗"]
    if 'past1' not in st.session_state:
        st.session_state['past1'] = ["hey"]
    if 'history1' not in st.session_state:
        st.session_state['history1'] = []


def display_chat_history1():
    # reply_container = st.container()
    # container = st.container()

    # with container:
    with st.form(key='my_form', clear_on_submit=False):
        user_input = st.text_input("Question:", placeholder="Your Query", key='input122')
        # import streamlit as st

        # user_input = st.chat_input("Say something")
        # if user_input:
        #     st.write(f"User has sent the following prompt: {user_input}")
        configure = st.sidebar.button('configure')
        submit_button = st.form_submit_button(label='Send')
        st.session_state['past1'].append(user_input)

    if (submit_button and user_input) or (configure and user_input):
        output = conversation_chat1(user_input)
        # st.write(output)
    
        
        # st.session_state['past1'].append(user_input)
        st.session_state['generated1'].append(output)
        
        
            

        # st.text_area("Chatbot:", value=st.session_state['generated1'][-1],height=int(len(st.session_state['generated1'][-1])/16), max_chars=None, key="awe")
        st.write(st.session_state['generated1'][-1])
        # st.text_area("Chatbot:", value=st.session_state['generated1'][-1], height=None, max_chars=None, key="awe")
        # st.write(st.session_state['depth'])

initialize_session_state1()
# Display chat history
display_chat_history1()