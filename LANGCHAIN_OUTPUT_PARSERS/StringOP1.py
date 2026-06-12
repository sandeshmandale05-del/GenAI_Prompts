from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser

key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    temperature= 0.7,
    task="Text-Generation",
    huggingfacehub_api_token = key
)

model=ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="write a detailed report on{topic}",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="write a five line summery on the following text /n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Black hole"})
print(result)