from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
key=os.getenv("GOOGLE_API_KEY")

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=key
)

chat_history = [
    SystemMessage(content= " You are a helpful AI assistant")
]
while True:
    user_input = input("You : ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI : "  , result.content)
    
print(chat_history)    