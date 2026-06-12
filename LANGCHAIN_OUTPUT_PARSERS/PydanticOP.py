from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
load_dotenv()

key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    temperature= 0.7,
    task="Text-Generation",
    huggingfacehub_api_token = key
)

model=ChatHuggingFace(llm=llm)

class person(BaseModel):
    
    name : str = Field(description= " name of the person")
    age : int = Field(description="age of the person ")
    city : str = Field(description="name of the city person belongs to ")
    
parser = PydanticOutputParser(pydantic_object=person)

template =PromptTemplate(
    template="generate the name , age and cuty of the fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# chain = template | model | parser
# result=chain.invoke({"place": "indian"})
# print(result)

prompt=template.invoke({"place":"indian"})

result = model.invoke(prompt)

final_result =parser.parse(result.content)

print(final_result)