


import streamlit as st
from fpdf import FPDF
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph
import json
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, END
from langchain.tools.render import format_tool_to_openai_function

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
from langchain_openai import ChatOpenAI
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

st.image('Amrita-vishwa-vidyapeetham-logo.svg.png')
st.title("     WELCOME TO SARASWATI AI")
    

url = requests.get( 
"https://lottie.host/57f3be46-32cc-44b8-9932-e9d84aa9dbd1/xRKnHUCJA1.json")
# Creating a blank dictionary to store JSON file, 
# as their structure is similar to Python Dictionary 
url_json = dict() 

if url.status_code == 200: 
    url_json = url.json() 
else: 
    print("Error in the URL") 
with st.sidebar:
    st_lottie(url_json,reverse=True, 
        # height and width of animation 
        height=400,   
        width=300, 
        # speed of animation 
        speed=1,   
        # means the animation will run forever like a gif, and not as a still image 
        loop=True,   
        # quality of elements used in the animation, other values are "low" and "medium" 
        quality='high', 
        # THis is just to uniquely identify the animation 
        key='Car' )



# st.subheader("Unleash Your Learning Potential with saraswati")
st.write("Saraswati your one-stop shop for a smarter, more personalized learning experience!")
st.subheader("Here's what makes special:")
st.subheader("Chatbot Guru :")
st.write("Stuck on a problem buried in a PDF?  Our friendly AI chatbot can answer your questions directly from the text! Just upload your document and let the magic happen. ✨")
st.subheader("Path Planner")
st.write("Feeling lost in your studies?   Our Path Planner analyzes your strengths and weaknesses to create a customized learning roadmap that guides you towards success.  🤖")
st.subheader("AI Tutor at Your Fingertips")
st.write("Get instant feedback and explanations from our intelligent AI tutor.    Whether you need a concept clarified or practice solving problems, EduAI is there to support you every step of the way.  📖")
st.subheader("Learning has never been so easy and efficient! ")

