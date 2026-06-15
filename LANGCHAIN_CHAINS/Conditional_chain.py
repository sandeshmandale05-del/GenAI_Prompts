from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from typing import Literal 
from pydantic import BaseModel , Field
from langchain_core.runnables import RunnableParallel , RunnableLambda , RunnableBranch
import os 
load_dotenv()
key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    temperature= 0.7,
    task="Text-Generation",
    huggingfacehub_api_token = key
)

model=ChatHuggingFace(llm=llm)

# key=os.getenv("GOOGLE_API_KEY")

# model=ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature = 0.7,
#     google_api_key= key
# )

class feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser = StrOutputParser()

parser2=PydanticOutputParser(pydantic_object=feedback)

prompt1=PromptTemplate(
    template="classify the sentiment of the following feedback text into positive or negative \n {feedback}  \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction":parser2.get_format_instructions()
        }
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = "write an appropriate responce to this  positive feedback \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template = "write an appropriate responce to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback" , "this is very terrable smart phone"}))

chain.get_graph().print_ascii()


