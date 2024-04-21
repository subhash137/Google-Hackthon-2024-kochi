


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
# from langchain_openai import ChatOpenAI
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

# tab1, tab2, tab3, tab4 = st.tabs(["Path Planner", "Tutor", "Chatbot","Story Teller"])

# tab_names = ["Home","Path Planner", "Tutor", "Chatbot", "Story Teller"]
# selected_tab_index = st.session_state.get('selected_tab_index', 0)
# # selected_tab = st.tabs(tab_names, selected_tab_index)

# # # Update selected tab index in session state
# # st.session_state.selected_tab_index = tab_names.index(selected_tab)

# selected_tab_name = st.sidebar.selectbox("Select Tab", tab_names, index=selected_tab_index)

# # Update selected tab index in session state
# st.session_state.selected_tab_index = tab_names.index(selected_tab_name)
# Display tab content based on selection

from st_pages import Page, add_page_title, show_pages





show_pages(
    [
        Page("streamlit_app.py", "Home", "🏠"),
        # Can use :<icon-name>: or the actual icon
        Page("chatbot.py", "Chatbot Guru", "🤖"),
        # The pages appear in the order you pass them
        Page("Tutor.py", "AI Tutor", "🧑‍🏫"),
        Page("Path_planner.py", "Path Planning", "✏"),
        Page("Quiz.py", "Quiz", "📖"),
        Page("oral_session.py", "Interview", "🎙️"),
        
    ]
)

add_page_title()  # Optional method to add title and icon to current page
