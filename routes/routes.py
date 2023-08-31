from fastapi import APIRouter, UploadFile,File
from uploadservice import UploadService
from models.model import Prompt
import numpy as np
import pinecone
import openai
import os
import json

from redis_client import RedisClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

routes=APIRouter()


#endopoints

@routes.get("/")
def welcome():
    return "Bienvenido a IADRES"
#Endpoint en desuso
# # @routes.post("/load-file")  
# # async def generate_response_pdf(prompt:str, file: UploadFile = File(...)):
#     try:
#         #Carga y embedding del archivo
#         data=await UploadService().analize_file(file)
#         print(data)
#         #Responde si se envia el similarity search aquí dentro      

#         chain = load_qa_chain(OpenAI(), chain_type="stuff")  
       
#         docs = data.similarity_search(prompt)
#         return chain.run(input_documents=docs, question=prompt)

    
#     except Exception as e:
#         print("Error:", str(e))

#endpoint que carga archivo, generar embeddings y subir a PineCone
@routes.post("/load-data")
async def load_pdf(file: UploadFile = File(...)):
    await UploadService().embedding_text(file)

#endpoint que ejecuta prompts sobre archivo pre-cargado
@routes.post("/send-prompt")
async def send_prompt(prompt: Prompt, index_name: Prompt):
    pinecone.init(api_key=os.environ.get('PINECONE_APY_KEY'), environment=os.environ.get('PINECONE_ENV'))

    # Crear una instancia del índice
    text_field = "text"

    # switch back to normal index for langchain
    index = pinecone.Index(index_name.text)
    embeddings=OpenAIEmbeddings()
    vectorstore = Pinecone(
        index, embeddings.embed_query, text_field
    )
        
    # Texto de consulta
    query= f"{prompt.text} respuesta en español"

    # completion llm
    llm = ChatOpenAI(
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
        model_name='gpt-3.5-turbo',
        temperature=0.0,
    )

    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
    )
    answer= qa_with_sources(query)
    return answer
    
    
#endpoint para cargar archivo, generar emneddings y subir a Redis
# @routes.post("/load-file-redis")
# async def load_pdf(file: UploadFile = File(...)):
#     await UploadService().generate_embeddings(file)


# @routes.post("/send-prompt-redis")
# async def send_prompt(prompt: Prompt, index_name: Prompt):
#     redis_client=RedisClient(os.environ.get('REDIS_HOST'), os.environ.get('REDIS_PORT'))  
#     # switch back to normal index for langchain
#     key = index_name.text
#     embeddings=OpenAIEmbeddings()
    
#     embeddings_data = redis_client.get(key)
#     embeddings_list = json.loads(embeddings_data)
#     print (embeddings_list[0])
        
#     # Texto de consulta
#     query= f"{prompt.text} respuesta en español"

#     # completion llm
#     llm = ChatOpenAI(
#         openai_api_key=os.environ.get('OPENAI_API_KEY'),
#         model_name='gpt-3.5-turbo',
#         temperature=0.0,
#     )

#     qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=embeddings_list.as_retriever()
#     )
#     answer= qa_with_sources(query)
#     return answer
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
        """#Responde si se envia el similarity search aquí dentro    
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI

        chain = load_qa_chain(OpenAI(), chain_type="stuff")  
       
        docs = data.similarity_search(prompt)
        return chain.run(input_documents=docs, question=prompt)

        #collection_embeddings.insert_one({"filename":file.filename, "embeddings":data})
        """
    
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

@routes.post("/load-pdf")  
async def generate_response_pdf(file: UploadFile = File(...)):
    try:
        #Carga y embedding del archivo
        data=await UploadService().embedding_file(file)
        
        """#Responde si se envia el similarity search aquí dentro    
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI

        chain = load_qa_chain(OpenAI(), chain_type="stuff")  
       
        docs = data.similarity_search(prompt)
        return chain.run(input_documents=docs, question=prompt)

        #collection_embeddings.insert_one({"filename":file.filename, "embeddings":data})
        """
    
    except Exception as e:
        print("Error:", str(e))
