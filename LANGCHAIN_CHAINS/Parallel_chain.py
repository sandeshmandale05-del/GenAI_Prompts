from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import os 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel
load_dotenv()

key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
key2=os.getenv("GOOGLE_API_KEY")
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    temperature= 0.7,
    task="Text-Generation",
    huggingfacehub_api_token = key
)

model=ChatHuggingFace(llm=llm)
model2=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=key2
)
Prompt1=PromptTemplate(
    template="Generate the detailed Study on {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)
parser=StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": Prompt1 | model2 | parser ,
    "quiz":  prompt2 | model | parser 
}) 

merge_chain = prompt3 | model2 | parser 

chain = parallel_chain | merge_chain | parser 
try:
    result=chain.invoke({"topic" , "black hole"})
    print(result)
    
except Exception as e: 
    print("error" , e )   
    
     
chain.get_graph().print_ascii()

