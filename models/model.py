from pydantic import BaseModel,Field

class Prompt(BaseModel):
    text:str= Field(min_length=10,max_length=100),
    