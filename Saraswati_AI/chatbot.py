


import streamlit as st
from fpdf import FPDF
import os
import json

import json

from langchain_chroma import Chroma

import json

import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory


import json 

from langchain_core.messages import BaseMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from langchain.prompts import (
ChatPromptTemplate,
MessagesPlaceholder,

)
from langchain_google_vertexai import VertexAIEmbeddings
from streamlit_lottie import st_lottie
# llm = ChatOpenAI(api_key="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")
from langchain_google_vertexai import VertexAI
import vertexai

vertexai.init(project="saraswati-ai", location="us-central1")

llm = VertexAI(model_name="gemini-pro")

loader = DirectoryLoader('data/',glob="*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()

#split text into chunks
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

#create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device':"cpu"})

vector_store = Chroma.from_documents(text_chunks, embeddings)
# vector_store = FAISS.from_documents(text_chunks,embeddings)

# from langchain_community.llms import Ollama
# llm = Ollama(model='llama2')
# prompt = "You are AI assistant chatbot . You can use emoji's if you want but don't use everytime . GIve the concise content. "
prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are AI assistant chatbot . You can use emoji's if you want but don't use everytime . GIve the concise content. "
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
# llm1 = prompt | llm

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                                condense_question_prompt=prompt,
                                                retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                                memory=memory)

st.title("ChatBot Guru 🤖")
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Your Query", key='input12')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()