import json


import streamlit as st
# from get_quiz import get_quiz_from_topic

# Input box to enter the topic of the quiz
import json
from typing import Dict

from langchain_google_vertexai import VertexAI
import vertexai
from langchain_google_vertexai import ChatVertexAI

vertexai.init(project="saraswati-ai", location="us-central1")

llm = VertexAI(model_name="gemini-pro")

template1 = """
.
You are an AI Quiz Master.
Previous conversation:
{chat_history}


New human question: {question}
Response:"""



from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template=template1,input_variables=['question','chat_history','depth','learning','tone','reasoning']
)
chain = prompt | llm


chat_history = [
    {
        "role": "system",
        "content": "You are a REST API server with an endpoint /generate-random-question/:topic, which generates unique random quiz question in json data.",
    },
    {"role": "user", "content": "GET /generate-random-question/devops"},
    {
        "role": "assistant",
        "content": '\n\n{\n    "question": "What is the difference between Docker and Kubernetes?",\n    "options": ["Docker is a containerization platform whereas Kubernetes is a container orchestration platform", " Kubernetes is a containerization platform whereas Docker is a container orchestration platform", "Both are containerization platforms", "Neither are containerization platforms"],\n    "answer": "Docker is a containerization platform whereas Kubernetes is a container orchestration platform",\n    "explanation": "Docker helps you create, deploy, and run applications within containers, while Kubernetes helps you manage collections of containers, automating their deployment, scaling, and more."\n}',
    },
    {"role": "user", "content": "GET /generate-random-question/jenkins"},
    {
        "role": "assistant",
        "content": '\n\n{\n    "question": "What is Jenkins?",\n    "options": ["A continuous integration server", "A database management system", "A programming language", "An operating system"],\n    "answer": "A continuous integration server",\n    "explanation": "Jenkins is an open source automation server that helps to automate parts of the software development process such as building, testing, and deploying code."\n}',
    },
]


def get_quiz_from_topic(topic: str) -> Dict[str, str]:
    global chat_history
    # openai.api_key = api_key
    current_chat = chat_history[:]
    current_user_message = {
        "role": "user",
        "content": f"GET /generate-random-question/{topic}",
    }
    current_chat.append(current_user_message)
    chat_history.append(current_user_message)

    response = chain.invoke({'question':current_chat,'chat_history':chat_history})
    quiz = response
    current_assistent_message = {"role": "assistant", "content": quiz}
    chat_history.append(current_assistent_message)
    # print(f"Response:\n{quiz}")
    import re
    quiz = re.sub(r'```json|```', '', quiz)

    # print(quiz.strip())
    
    # quiz.replace
    
    json_string = json.loads(quiz)
    # json_string.strip("```")
    # print(quiz)
    return json_string

topic = st.sidebar.text_input(
    "To change topic just enter in below. From next new quiz question the topic entered here will be used.",
    value="devops",
)

# api_key = st.sidebar.text_input("OpenAI API key", type="password").strip()

# Initialize session state variables if they don't exist yet
if "current_question" not in st.session_state:
    st.session_state.answers = {}
    st.session_state.current_question = 0
    st.session_state.questions = []
    st.session_state.right_answers = 0
    st.session_state.wrong_answers = 0


# Define a function to display the current question and options
def display_question():
    # Handle first case
    if len(st.session_state.questions) == 0:
        
        first_question = get_quiz_from_topic(topic)
        
        return st.session_state.questions.append(first_question)

    # Disable the submit button if the user has already answered this question
    submit_button_disabled = st.session_state.current_question in st.session_state.answers

    # Get the current question from the questions list
    question = st.session_state.questions[st.session_state.current_question]

    # Display the question prompt
    st.write(f"{st.session_state.current_question + 1}. {question['question']}")

    # Use an empty placeholder to display the radio button options
    options = st.empty()

    # Display the radio button options and wait for the user to select an answer
    user_answer = options.radio("Your answer:", question["options"], key=st.session_state.current_question)

    # Display the submit button and disable it if necessary
    submit_button = st.button("Submit", disabled=submit_button_disabled)

    # If the user has already answered this question, display their previous answer
    if st.session_state.current_question in st.session_state.answers:
        index = st.session_state.answers[st.session_state.current_question]
        options.radio(
            "Your answer:",
            question["options"],
            key=float(st.session_state.current_question),
            index=index,
        )

    # If the user clicks the submit button, check their answer and show the explanation
    if submit_button:
        # Record the user's answer in the session state
        st.session_state.answers[st.session_state.current_question] = question["options"].index(user_answer)

        # Check if the user's answer is correct and update the score
        if user_answer == question["answer"]:
            st.write("Correct!")
            st.session_state.right_answers += 1
        else:
            st.write(f"Sorry, the correct answer was {question['answer']}.")
            st.session_state.wrong_answers += 1

        # Show an expander with the explanation of the correct answer
        with st.expander("Explanation"):
            st.write(question["explanation"])

    # Display the current score
    st.write(f"Right answers: {st.session_state.right_answers}")
    st.write(f"Wrong answers: {st.session_state.wrong_answers}")


# Define a function to go to the next question
def next_question():
    # Move to the next question in the questions list
    st.session_state.current_question += 1

    # If we've reached the end of the questions list, get a new question
    if st.session_state.current_question > len(st.session_state.questions) - 1:
        
        next_question = get_quiz_from_topic(topic)
       
        return st.session_state.questions.append(next_question)


# Define a function to go to the previous question
def prev_question():
    # Move to the previous question in the questions list
    if st.session_state.current_question > 0:
        st.session_state.current_question -= 1
        st.session_state.explanation = None


# Create a 3-column layout for the Prev/Next buttons and the question display
st.header("Quiz")

col1, col2, col3 = st.columns([1, 6, 1])

# Add a Prev button to the left column that goes to the previous question
with col1:
    if col1.button("Prev"):
        prev_question()

# Add a Next button to the right column that goes to the next question
with col3:
    if col3.button("Next"):
        next_question()

# Display the actual quiz question
with col2:
    display_question()

# Add download buttons to sidebar which download current questions
download_button = st.sidebar.download_button(
    "Download Quiz Data",
    data=json.dumps(st.session_state.questions, indent=4),
    file_name="quiz_session.json",
    mime="application/json",
)
