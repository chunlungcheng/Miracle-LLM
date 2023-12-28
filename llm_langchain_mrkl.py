import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.llms import OpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain, LLMChain
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import streamlit as st

# load_dotenv()

# Set up streamlit UI
st.set_page_config(page_title="OpenAI x LangChain x MRKL", layout="wide", initial_sidebar_state="collapsed")

st.title("OpenAI x LangChain x MRKL")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type= "password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Type in your question here"):
    st.session_state.messages.append({"role": "user", "content":prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please enter a valid OpenAI API Key")
        st.stop()
    
    # Set up tools(MRKL)
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]

    mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = mrkl.run(st.session_state.messages, callbacks=[cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)