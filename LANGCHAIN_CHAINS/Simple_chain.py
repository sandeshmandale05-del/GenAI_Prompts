from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import os 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    temperature= 0.7,
    task="Text-Generation",
    huggingfacehub_api_token = key
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="give me detailed explaination on the topic \n {text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="give me 5 line summary on the {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | model |parser |  prompt2 |model |parser
user_input=input("Enter the name you want to study about : ")

result = chain.invoke({" text" , user_input})

print(result)
chain.get_graph().print_ascii()