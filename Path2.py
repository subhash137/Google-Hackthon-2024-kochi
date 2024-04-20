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
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.agents import AgentActionMessageLog
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
                "Dont ask the user loike this in resources. Don't ask for sepcific answer or i don't know just provide answer (path) - (insert YouTube link)"
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="input"),
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
          func = lambda topic_name: tool1.run(topic_name),
          description = "use when you want to give youtube links just input content ot topic name and size number of links you want. ",
      ),
      

        ]
prompt1 = hub.pull("hwchase17/react")
tool_executor = ToolExecutor(tools)
# functions = [format_tool_to_openai_function(t) for t in tools]
agent_runnable = create_react_agent(llm,tools,prompt1)
model = prompt | agent_runnable

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    return_direct: bool
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    
tool_executor = ToolExecutor(tools)



def run_agent(state):
    """
    #if you want to better manages intermediate steps
    inputs = state.copy()
    if len(inputs['intermediate_steps']) > 5:
        inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
    """
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome} 


def execute_tools(state):

    messages = [state['agent_outcome'] ]
    last_message = messages[-1]
    ######### human in the loop ###########   
    # human input y/n 
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    # state_action = state['agent_outcome']
    # human_key = input(f"[y/n] continue with: {state_action}?")
    # if human_key == "n":
    #     raise ValueError
    
    tool_name = last_message.tool
    arguments = last_message
    if tool_name == "Search"  or tool_name == "Youtube Link provider":
        
        if "return_direct" in arguments:
            del arguments["return_direct"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input= last_message.tool_input,
    )
    response = tool_executor.invoke(action)
    return {"intermediate_steps": [(state['agent_outcome'],response)]}

def should_continue(state):

            messages = [state['agent_outcome'] ] 
            last_message = messages[-1]
            if "Action" not in last_message.log:
                return "end"
            else:
                arguments = state["return_direct"]
                if arguments is True:
                    return "final"
                else:
                    return "continue"


def first_agent(inputs):
            action = AgentActionMessageLog(
            tool="Search",
            tool_input=inputs["input"],
            log="",
            message_log=[]
            )
            return {"agent_outcome": action}
        
workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.add_node("final", execute_tools)
# uncomment if you want to always calls a certain tool first
# workflow.add_node("first_agent", first_agent)


workflow.set_entry_point("agent")
# uncomment if you want to always calls a certain tool first
# workflow.set_entry_point("first_agent")

workflow.add_conditional_edges(

    "agent",
    should_continue,

    {
        "continue": "action",
        "final": "final",
        "end": END
    }
)


workflow.add_edge('action', 'agent')
workflow.add_edge('final', END)
# uncomment if you want to always calls a certain tool first
# workflow.add_edge('first_agent', 'action')
app = workflow.compile()
result = app.invoke({"input": "machine learning", "chat_history": [], "return_direct": False})

print(result["agent_outcome"].return_values["output"])



