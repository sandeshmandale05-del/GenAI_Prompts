# this langchain.output_parsers class works only in old models of langchain but now in latest models it cant 

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain_core.output_parsers import StrOutputParser , JsonOutputParser

key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    temperature= 0.7,
    task="Text-Generation",
    huggingfacehub_api_token = key
)

model=ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = "Fact_1" , description = "Fact 1 about the topic"),
    ResponseSchema(name = "Fact_2" , description = "Fact 2 about the topic"),
    ResponseSchema(name = "Fact_3" , description = "Fact 3 about the topic")
]

parser = StructuredOutputParsers.from_response_schemas(schema)

template=PromptTemplate(
    template = " Give 3 facts about {topic}  \n {Format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)


# chain = template | model | parser
# result=chain.invoke({"topic": "Black hole"})
# print(result)


prompt = template.invoke({"topic": "Black hole"})

result = model.invoke(prompt)
final_result =parser.parse(result.content)
print(final_result)
