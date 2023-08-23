from fastapi import APIRouter, UploadFile,File
from uploadservice import UploadService
from models.model import Prompt
from config.db import collection_embeddings
from bson import ObjectId
import numpy as np

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

routes=APIRouter()


#endopoints

@routes.get("/juridica")
def welcome():
    return "Bienvenido a IADRES"

@routes.post("/load-file")  
async def generate_response_pdf(file: UploadFile = File(...)):
    try:
        #Carga y embedding del archivo
        data=await UploadService().analize_file(file)
        
    
    except Exception as e:
        print("Error:", str(e))
    

@routes.post("/test-prompt")
async def test_prompt(prompt:Prompt):
    try:

        query=prompt.text

        search_text = "64e62bd92ceaa63e203581a3"

        # Busca el documento en la base de datos
        result = collection_embeddings.find_one({"_id": ObjectId(search_text)})

        if result:
            embedding = result["text"]
            embedding_array = np.array(embedding)  # Convierte la lista en un array NumPy
            print(embedding_array)
            print("Embedding encontrado:")
           
        else:
            print("Embedding no encontrado para el texto:", search_text)
         
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI

        chain = load_qa_chain(OpenAI(), chain_type="stuff")        
        docs = data.similarity_search(query)
        return chain.run(input_documents=docs, question=query)
    
    except Exception as e:
        print("Error:", str(e))
