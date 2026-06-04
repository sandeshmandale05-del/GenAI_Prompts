from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt
load_dotenv()

key=os.getenv("GOOGLE_API_KEY")
model=GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=key
) 

paper_input = st.selectbox( "Select Research Paper Name",
                           ["Attention Is All You Need",
                            "BERT: Pre-training of Deep Bidirectional Transformers", 
                            "GPT-3: Language Models are Few-Shot Learners", 
                            "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", 
                           ["Beginner-Friendly", 
                            "Technical", 
                            "CodeOriented", 
                            "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", 
                            ["Short (1-2 paragraphs)", 
                             "Medium (3-5 paragraphs)", 
                             "Long (detailed explanation)"] )

template= load_prompt("template.json")

# prompt = template.invoke({
#     "paper_input": paper_input,
#     "style_input": style_input,
#     "length_input":length_input
# })

# st.header("research Tool")

# if st.button("Summerize"):
#     result = model.invoke(prompt)
#     st.write(result)

if st.button("Summerize"):
    chain = template | model
    result = chain.invoke({
        "paper_input":paper_input,
        "style_input":style_input,
        "length_input":length_input
    })
    st.write(result)