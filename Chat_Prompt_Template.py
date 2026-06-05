from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage , SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()
chat_template = ChatPromptTemplate([
    ("system", " You are a {domain} expert"),
    ("human", " tell mi about {topic}")
    # SystemMessage(content = " You are a {domain} expert"),
    # HumanMessage(content = "tell mi about {topic}")
])

prompt = chat_template.invoke({"domain": " AI" , "topic": " AI"})
print(prompt) 