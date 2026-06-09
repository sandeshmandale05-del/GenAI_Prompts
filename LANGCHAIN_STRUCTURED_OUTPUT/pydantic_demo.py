from typing import TypedDict , Optional
from pydantic import BaseModel , EmailStr , Field

class Students(BaseModel):
    name : str = "Gaurav"
    age : Optional[int] =None
    collage : str = " Dyanesh Mahavidhyalaya Nawargaon"
    email : EmailStr
    cgpa :float =Field(gt=0 , lt=10 ,default=7)
    
new_Student = {"age" : 21 , "email" : "abc@ycce.in" , "cgpa" : 9.6  }    

Student = Students(**new_Student)

print(Student)