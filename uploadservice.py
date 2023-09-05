# python imports
import os
import json
from os import getcwd
import pandas as pd
# FastAPI imports

from PyPDF2 import PdfReader
from fastapi import UploadFile,File
from langchain.vectorstores.redis import Redis
from redis_client import RedisClient
from dotenv import load_dotenv

import openai,numpy as np,time
import pinecone 
import matplotlib.pyplot as plt

from openai.embeddings_utils import get_embedding
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter as RC
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone, FAISS
from models.model import Prompt

# Cargar variables de entorno desde el archivo .env
load_dotenv()

class UploadService():

    openai.api_key=os.environ.get('OPENAI_API_KEY')
    openai.organization=os.environ.get('OPENAI_ORG_ID')

    def __init__(self):
        # Path of files to upload
        self.path = getcwd()
        self.resultList = []
    

    async def analize_file(self, file: UploadFile = File):
        try:
             with open(os.path.join(self.path, file.filename), "wb") as f:
                content = await file.read()
                f.write(content)
                f.close()
                data=f"./{file.filename}"
                reader = PdfReader(data)
                #print(reader)
                # read data from the file and put them into a variable called raw_text
                raw_text = ''
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        raw_text += text

                #print(raw_text[:100])
                # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

                text_splitter = CharacterTextSplitter(        
                    separator = "\n",
                    chunk_size = 1000,
                    chunk_overlap  = 200,
                    length_function = len,
                )
                texts = text_splitter.split_text(raw_text)

                print(len(texts))
                #print(texts)

                embeddings = OpenAIEmbeddings()

                #prueba pinecone
                
                docsearch = FAISS.from_texts(texts, embeddings)
                return docsearch
     

        except Exception as e:
            print("Error:", str(e))

    async def embedding_text(self, file: UploadFile = File):
            try:
                #Leer archivo
                from langchain.document_loaders import PyPDFLoader
                with open(os.path.join(self.path, file.filename), "wb") as f:
                    content = await file.read()
                    f.write(content)
                    f.close()
                    data=f"./{file.filename}"
                    reader = PyPDFLoader(data)
                    fileload = reader.load()
                                       
                    #fragmentar los textos
                    text_splitter = RC(        
                        chunk_size = 1000,
                        chunk_overlap  = 50,
                        length_function = len,
                    )
                    texts = text_splitter.split_documents(fileload)

                    print(len(texts))
                    print(texts[0])

                    embeddings = OpenAIEmbeddings()
                   
                    #Enviar vectores a index pinecone
                    pinecone.init(api_key=os.environ.get('PINECONE_APY_KEY'),environment=os.environ.get('PINECONE_ENV'))
                    
                    nombre, extension = os.path.splitext(file.filename)
                    index_n=nombre.lower()

                    active_indexes = pinecone.list_indexes()
                    if active_indexes   == []:
                        print("La lista está vacía")
                        if index_n not in pinecone.list_indexes():
                            print(f"Creando el índice {index_n} ...")
                            pinecone.create_index(index_n,dimension=1536,metric='cosine')
                            print("Done!")
                            vector_store=Pinecone.from_documents(texts,embeddings,index_name=index_n)

                        else:
                            print(f"El índice {index_n} ya existe puedes ejecutar tus prompts")
                        
                    else:
                        index_delete=pinecone.list_indexes()[0]
                        pinecone.delete_index(index_delete)
                        print(f"Eliminando el índice {active_indexes[0]}...")
                        print(f"Creando el índice {index_n} ...")
                        pinecone.create_index(index_n,dimension=1536,metric='cosine')
                        print("Done!")
                        
                    active_indexes = pinecone.list_indexes()
                    if active_indexes   == []:
                        print("La lista está vacía")
                        if index_n not in pinecone.list_indexes():
                            print(f"Creando el índice {index_n} ...")
                            pinecone.create_index(index_n,dimension=1536,metric='cosine')
                            print("Done!")
                            vector_store=Pinecone.from_documents(texts,embeddings,index_name=index_n)

                        else:
                            print(f"El índice {index_n} ya existe puedes ejecutar tus prompts")
                        
                    else:
                        index_delete=pinecone.list_indexes()[0]
                        pinecone.delete_index(index_delete)
                        print(f"Eliminando el índice {active_indexes[0]}...")
                        print(f"Creando el índice {index_n} ...")
                        pinecone.create_index(index_n,dimension=1536,metric='cosine')
                        print("Done!")
                        
                    vector_store=Pinecone.from_documents(texts,embeddings,index_name=index_n)

                    return index_n

      
                       
                    
             
                        
                       
                                       
            
            except Exception as e:
                print("Error:", str(e))

    async def generate_embeddings(self, file:UploadFile=File):
        try:
                #Leer archivo
                from langchain.document_loaders import PyPDFLoader
                with open(os.path.join(self.path, file.filename), "wb") as f:
                    content = await file.read()
                    f.write(content)
                    f.close()
                    data=f"./{file.filename}"
                    reader = PyPDFLoader(data)
                    fileload = reader.load()
                                       
                    #fragmentar los textos
                    text_splitter = RC(        
                        chunk_size = 1000,
                        chunk_overlap  = 50,
                        length_function = len,
                    )
                    docs = text_splitter.split_documents(fileload)
                    
                    print(len(docs))
                    print(docs[0])

                    docs = [str(i.page_content) for i in docs] #Lista de parrafos
                    parrafos = pd.DataFrame(docs, columns=["texto"])
                    parrafos['Embedding'] = parrafos["texto"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002')) # Nueva columna con los embeddings de los 
                    parrafos.to_csv('MTG.csv')
                    print(parrafos)

                    embeddings_list = []
                    for idx, row in parrafos.iterrows():
                        embedding = row["Embedding"]
                        embeddings_list.append({"text": row["texto"], "embedding": embedding})
                    

                    # embeddings = OpenAIEmbeddings()

                    nombre, extension = os.path.splitext(file.filename)
                    key=nombre.lower()

                    redis_client=RedisClient(os.environ.get('REDIS_HOST'), os.environ.get('REDIS_PORT'))   

                    if not redis_client.exists(key):         
                        time.sleep(1)
                        print("Creando embedding para: "+key)
                        redis_client.set(key, json.dumps(embeddings_list))  

                    else:
                        embeddings_data = redis_client.get(key)
                        embeddings_list = json.loads(embeddings_data)

                    return embeddings_list           
                  
            
        except Exception as e:
            print("Error:", str(e))
       