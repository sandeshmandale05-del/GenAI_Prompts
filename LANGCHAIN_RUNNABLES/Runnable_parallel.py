from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnablePassthrough , RunnableParallel
import os

load_dotenv()

key=os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key = key
)

prompt1=PromptTemplate(
    template="generate a tweet about \n {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

prompt2=PromptTemplate(
    template= " generate a linked in post \n {topic}",
    input_variables=["topic"]
)

parallel_chain=RunnableParallel({
    "tweet" :RunnableSequence(prompt1 , model , parser),
    "linkedin":RunnableSequence(prompt2 , model , parser)
})

chain =parallel_chain.invoke({
    "topic" : "AI"
})

print(chain)