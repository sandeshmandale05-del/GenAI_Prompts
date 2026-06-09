from typing import TypedDict

class person(TypedDict):
    
    name : str
    age : int
    
name_person : person = {
    "name" : "Sumit",
    "age" : 21
}    

print(name_person)