from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda , RunnableParallel , RunnableSequence , RunnablePassthrough
import os
load_dotenv()

key =os.getenv("GOOGLE_API_KEY")

model=ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.7,
    google_api_key = key
)

def word_count(text):
    return len(text.split())

prompt1=PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

joke_chain_gen = RunnableSequence(prompt1 , model , parser)

parallel_chain =RunnableParallel({
    "joke" : RunnablePassthrough(),
    "word_count":RunnableLambda(word_count)
})

final_chain =RunnableSequence(joke_chain_gen , parallel_chain)

result=final_chain.invoke({"topic":"AI"})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)