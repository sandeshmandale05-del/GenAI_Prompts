from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

key=os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key = key
)

parser=StrOutputParser()

prompt1 =PromptTemplate(
    template="Write a joke about topic \n {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Give the explaination about the joke \n {text}",
    input_variables=["text"]
)

sequence_ren = RunnableSequence(prompt1 , model , parser , prompt2 , model , parser)
result=sequence_ren.invoke({
    "topic":"Gold"
})

print(result)
