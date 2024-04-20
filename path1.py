from langchain.agents import Tool, create_react_agent

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
from langchain import hub
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

tool1 = YouTubeSearchTool()

@tool
def youtube_tool(
    topic_name: Annotated[str, "JUst give the topic name so that it generates youtube link"],
    size1 : Annotated[str, "Also mention how many links you needed by default keep 3"]
):
    """Use this to run youtube tool. Return the links in a list format ."""
    try:
        ter = topic_name + "," + size1
        result = tool1.run(ter)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Stdout: {result}"

@tool
def search_tv(Input1:Annotated[str,"Here input the text you want to search in web."]):
    """   Here You will text to search for content and get information , website links so that to get info upto date"""
    rt = TavilySearchResults().run(Input1)
    return rt

tools1 = [TavilySearchResults(max_results=1), youtube_tool]


prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, You are a path planner.  Explanation should be about 3 pages long"
                "First with your knowledge explain step by step then use tools e with topics, subtopics , materials"
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK,  "
                "  Execute what you can to make progress."
                " If you have the final answer or deliverable,"
                "Any question user gave you like dish preparation or anything just create a roadmap"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                "You will prepare a summary , paths , resources ."
                "If you have no links don't give like this - [Some text- YouTube](insert YouTube link) just say don't know"
                "In resources section Giving  links from the tools is must and compulsary if not there just say I didn't found resources"
                "Use search tools only for links if you don't know then only search for content and use that content"
                " For resources provide the links from taviely search tool and youtube tool. these tools will be provided to you "
                "Dont ask the user loike this in resources - (insert YouTube link)"
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


system_message="You should explain from beginnner to advanced level . First summary , steps upto 2 pages content then resources section give web links and youtube links .Explain step by step . If you don't know use taviley search and create steps . provide the links of youtube and taviely search also  . Also mention  Time required Difficulty level ."
prompt = prompt.partial(system_message=system_message)
prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools1]))






model = llm

tools = [
      Tool(
          name = "Search",
          func=TavilySearchResults().run,
          description="useful for when you need to answer questions about current events and also provide website links and content",
      ),
      Tool(
          name = "Youtube Link provider",
          func = lambda topic_name,size: tool1.run(topic_name,",",size),
          description = "use when you want to give youtube links just input content ot topic name and size number of links you want. ",
      ),
      

        ]
prompt1 = hub.pull("hwchase17/react")
tool_executor = ToolExecutor(tools)
# functions = [format_tool_to_openai_function(t) for t in tools]
model = create_react_agent(llm,tools,prompt1)
model = prompt | model 
print(model.invoke({'messages':"Helo"}))
# model = prompt | model.bind_functions(functions)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]



# Define the function that determines whether to continue or not
# @st.cache_data
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
# @st.cache_data
def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
# @st.cache_data
def call_tool(state):
    messages = state['messages']
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}



# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge('action', 'agent')

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

@st.cache_data
def invoke_api1(topic12):
    inputs = {"messages": [HumanMessage(content=topic12)],"chat_history": []}
    response = app.invoke(inputs)
    # Call the API to get the response
    return response

from langchain_core.messages import HumanMessage



# Format headers
if 'answer1' not in st.session_state:
    st.session_state['answer1'] = []
if 'user1' not in st.session_state:
    st.session_state['user1'] = []
st.header("Path Planner")
topic12 = st.text_input("Enter topic name:")

    


    
@st.cache_data
def fr(topic12):
    if topic12:
        st.session_state['user1'].append(topic12)
        provided_text =  invoke_api1(st.session_state['user1'][-1])
        
        formatted_text = format_headers(provided_text)
        
        
        st.session_state['answer1'].append(formatted_text)
        
        st.markdown(st.session_state['answer1'][-1])
        
        # st.write(st.session_state['user1'])
fr(topic12)


