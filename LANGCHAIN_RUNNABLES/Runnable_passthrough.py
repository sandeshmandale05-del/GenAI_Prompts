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
    template="write a joke on the topic \n {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

prompt2=PromptTemplate(
    template="explain the joke \n {text}",
    input_variables=["text"]
)

joke_gen_chain = RunnableSequence(prompt1 , model , parser )

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "explaination" : RunnableSequence(prompt2 , model , parser)
})

final_chain = RunnableSequence(joke_gen_chain , parallel_chain)
print(final_chain.invoke({"topic" : "history"}))