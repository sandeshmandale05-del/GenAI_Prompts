from langchain_core.messages import SystemMessage , HumanMessage ,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
key =os.getenv("GOOGLE_API_KEY")
model= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=key
)

messages = [
    SystemMessage(content = " You are a helpful assistant "),
    HumanMessage(content = " Tell me about AI and ML ")
]

result =model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)