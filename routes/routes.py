from fastapi import APIRouter, UploadFile,File
from uploadservice import UploadService
from models.model import Prompt
from config.db import collection_embeddings
from bson import ObjectId
import numpy as np

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

routes=APIRouter()


#endopoints

@routes.get("/juridica")
def welcome():
    return "Bienvenido a IADRES"

@routes.post("/load-file")  
async def generate_response_pdf(prompt:str, file: UploadFile = File(...)):
    try:
        #Carga y embedding del archivo
        data=await UploadService().analize_file(file)
        #print(data)
        #Responde si se envia el similarity search aquí dentro    
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI

        chain = load_qa_chain(OpenAI(), chain_type="stuff")  
       
        docs = data.similarity_search(prompt)
        return chain.run(input_documents=docs, question=prompt)

        #collection_embeddings.insert_one({"filename":file.filename, "embeddings":data})
        
    
    except Exception as e:
        print("Error:", str(e))
    

@routes.post("/test-prompt")
async def test_prompt(prompt:Prompt):
    try:

        query=prompt.text
        #print("Embedding encontrado:")
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI

        chain = load_qa_chain(OpenAI(), chain_type="stuff")  
       
        docs = data.similarity_search(query)
        return chain.run(input_documents=docs, question=query)
        # Busca el documento en la base de datos
        result = collection_embeddings.find_one({"filename": "E54051222045753R001297402700.pdf"})
        #print(result["embeddings"])
        
        if result:
            
            #print("Embedding encontrado:")
            from langchain.chains.question_answering import load_qa_chain
            from langchain.llms import OpenAI

            chain = load_qa_chain(OpenAI(), chain_type="stuff")  
            embed=  result["embeddings"]   
            docs = embed.similarity_search(query)
            return chain.run(input_documents=docs, question=query)
                
           
        else:
            print("Embedding no encontrado para el texto:", result)
         
        
    
    except Exception as e:
        print("Error:", str(e))
