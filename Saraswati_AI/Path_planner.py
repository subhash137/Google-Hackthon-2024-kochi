from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI
import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
import streamlit as st



st.title("Path Planner")
vertexai.init(project="saraswati-ai", location="us-central1")
llm = ChatVertexAI(model_name="gemini-pro", convert_system_message_to_human=True)


# st.
# result = llm.invoke("Write a ballad about LangChain")
# print(result.content)


import streamlit as st
from fpdf import FPDF
import os
import json

from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph
import json


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

from langchain_google_vertexai import VertexAI
import vertexai
from langchain_google_vertexai import ChatVertexAI

vertexai.init(project="saraswati-ai", location="us-central1")

llm = VertexAI(model_name="gemini-pro")
# llm = ChatOpenAI(model='google/gemini-pro')
# llm = ChatVertexAI(model_name="gemini-pro")

def save_as_pdf(formatted_text):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size = 12)
        pdf.multi_cell(0, 10, txt = formatted_text)
        pdf_file = "output.pdf"
        pdf.output(pdf_file)
        return pdf_file
def format_headers(text):
    lines = text.split("\n")
    formatted_text = ""
    for line in lines:
        if line.startswith("#:"):
            formatted_text += f"# {line.replace('#', '')}\n"
        elif line.startswith("#"):
            formatted_text += f"# {line.replace('#', '')}\n"
        # elif line.startswith("Time Required:"):
        #     formatted_text += f"# {'Time Required:'}\n"
        # elif line.startswith("Difficulty Level:"):
        #     formatted_text += f"# {'Difficulty Level:'}\n"
        elif line.startswith("**"):
            formatted_text += f"### {line.replace('**', '')}\n"
        else:
            formatted_text += f"{line}\n"
    return formatted_text
    # return formatted_text
    

TAVILY_API_KEY="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"

# Format headers
if 'answer1' not in st.session_state:
    st.session_state['answer1'] = []
if 'user1' not in st.session_state:
    st.session_state['user1'] = []
st.header("Path Planner")
topic12 = st.text_input("Enter topic name:")

    

if 'linky' not in st.session_state:
    st.session_state['linky'] = []
if 'linkt' not in st.session_state:
    st.session_state['linkt'] = []


template1 = """     
                "You are a helpful AI assistant, You are a  course planner for user's input course.  Explanation should be about 3 pages long"
                " you also get taviely websearch content and links if provided - {taviely}"
                " you also get provide youtube links - {youtube}"
                "Don;t mention anything about yourself and just give the content based on user's question"
                User's input - {question}
                
            """


prompt = PromptTemplate(
    template=template1,input_variables=['question','taviely','youtube']
)

chain = prompt | llm


@st.cache_data
def invoke_api1(topic12):
    inputs = {"question": [HumanMessage(content=topic12)],'youtube':st.session_state['linky'][-1],'taviely':st.session_state['linkt'][-1]}
    response = chain.invoke(inputs)
    # Call the API to get the response
    return response

from langchain_core.messages import HumanMessage




    
@st.cache_data
def fr(topic12):
    if topic12:
        
        st.session_state['user1'].append(topic12)
        tool1 = YouTubeSearchTool()
        rtt = topic12 + "," + "3"
        re3  = tool1.run(rtt)
        st.session_state['linky'].append(re3)
        re4 = TavilySearchResults(max_results=3).run(topic12)
        st.session_state['linkt'].append(re4)
        
        provided_text =  invoke_api1(st.session_state['user1'][-1])
        
        formatted_text = format_headers(provided_text)
        
        
        st.session_state['answer1'].append(formatted_text)
        
        st.markdown(st.session_state['answer1'][-1])
        
        # st.write(st.session_state['user1'])
fr(topic12)
