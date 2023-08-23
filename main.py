import io
import os

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile,File

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from models.model import Prompt
from uploadservice import UploadService

# Get your API keys from openai, you will need to create an account. 
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview


app=FastAPI()
load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]


"""
query = "cual es el nombre completo del paciente"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents=docs, question=query))
"""

#endopoints
  

@app.post("/juridica")  
async def generate_response_pdf(file: UploadFile = File(...)):
    try:
        #Carga y embedding del archivo
        data=await UploadService().analize_file(file)
        #Captura el prompt
        query = "cual es el nombre del paciente"
         
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI

        chain = load_qa_chain(OpenAI(), chain_type="stuff")        
        docs = data.similarity_search(query)
        return chain.run(input_documents=docs, question=query)
    
    except Exception as e:
        print("Error:", str(e))

    
    
    



    
    

       
    