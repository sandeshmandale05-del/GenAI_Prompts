from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch , RunnableSequence , RunnableParallel , RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os 
load_dotenv()

key=os.getenv("GOOGLE_API_KEY")

model=ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.7,
    google_api_key = key
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a detailed report on {report}",
    input_variables=["topic"]
)

prompt2 =PromptTemplate(
    template = "summerize the following {text}",
    input_variables=['text']
)

text_gen_chain = RunnableSequence(prompt1 , model , parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>300 ,prompt2 | model | parser ),
    RunnablePassthrough()
)

final_chain = RunnableSequence(text_gen_chain , branch_chain)

print(final_chain.invoke({"topic" , "Russia vs ukraine"}))

