from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated , Optional , Literal
from pydantic import BaseModel , EmailStr , Field
load_dotenv()
import os

key=os.getenv("GOOGLE_API_KEY")

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=key
)

#schema
class Review(BaseModel):
    
    key_themes : list[str] = Field(description=" Write down all the key themes discussed in a review in a list. ")
    summary : str = Field(description=" return brief summary of the review")
    Sentiments :Literal["pos","neg"]=Field(description= " return sentiment of review either positive , negative or neutral ")
    pros : Optional[list[str]] =Field(default=None , description="return all the pros from a review in a list . ")
    cons :Optional[list[str]] =Field(default=None , description=" return all the cons from a review in a list .") 
    name:Optional[list[str]]=Field(default=None , description="write the name of the reviewer")
    

# class review(TypedDict):
#     key_themes : Annotated[list[str] , " Write down all the key themes discussed in a review in a list . "]
#     summary :Annotated[str , " return brief summary of the review"]
#     sentiments :Annotated[str , " return sentiment of review either positive , negative or neutral "]
#     pros : Annotated[Optional[list[str]], "return all the pros from a review in a list . "]
#     cons : Annotated[Optional[list[str]] , " return all the cons from a review in a list ."] 

Structured_model = model.with_structured_output(Review)


result = Structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors. 

review by nitesh singh .
""" )
print(result.name)
print(result)