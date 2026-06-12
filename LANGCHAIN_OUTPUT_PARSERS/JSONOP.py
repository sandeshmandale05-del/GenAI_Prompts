from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser

key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    temperature= 0.7,
    task="Text-Generation",
    huggingfacehub_api_token = key
)

model=ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template= PromptTemplate(
    template = "Give me the name age and city of the fictional person /n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction" : parser.get_format_instructions()}
)

# prompt=template.format()
# print(prompt)

# result=model.invoke(prompt)

chain = template | model | parser

result = chain.invoke({})

final_result =parser.parse(result.content)
# print(final_result)