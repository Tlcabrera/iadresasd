# python imports
import os
import re
from os import getcwd
import pandas as pd
# FastAPI imports

from PyPDF2 import PdfReader
from fastapi import UploadFile,File
from faiss import IndexFlatL2


import numpy as np
import pickle 

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter as RC
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from models.model import Prompt
from config.db import collection_embeddings


class UploadService():

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

                """
                # Crear una lista para guardar los datos de los embeddings
                embedding_data = []
                for text, embedding in zip(texts, docsearch.embeddings):
                    embedding_data.append((text, *embedding))  # Agregar text y embedding
                column_names = ["text"] + [f"embedding_{i}" for i in range(embeddings.embedding_size)]
                df = pd.DataFrame(embedding_data, columns=column_names)

                    # Ruta del archivo CSV
                csv_filename = "embeddings.csv"

                # Guardar el DataFrame en un archivo CSV
                df.to_csv(csv_filename, index=False, encoding="utf-8")

                print(f"Los embeddings se han guardado en {csv_filename}")
            """
           

                

        except Exception as e:
            print("Error:", str(e))


    async def embedding_file(self, file: UploadFile = File):
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

                text_splitter = RC(       
                    chunk_size = 1000,
                    chunk_overlap  = 50,
                    length_function = len,
                )
                texts = text_splitter.split_text(raw_text)

                print(len(texts))
                embeddings = OpenAIEmbeddings()
                docsearch = FAISS.from_texts(texts, embeddings)
                
                #print(texts[0])
  
                #Crear embeddings
                def creando_vectores(index_name):
                    import pinecone
                    from langchain.vectorstores import Pinecone
                    from langchain.embeddings.openai import OpenAIEmbeddings
                    
                    embeddings = OpenAIEmbeddings()
                    
                    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), 
                                environment=os.environ.get('PINECONE_ENV'))
                    
                    if index_name in pinecone.list_indexes():
                        print(f'El índice {index_name} ya existe. Cargando los embeddings ... ', end='')
                        vectores = Pinecone.from_existing_index(index_name, embeddings)
                        print('Ok')
                    else:
                        print(f'Creando el índice {index_name} y los embeddings ...', end='')
                        pinecone.create_index(index_name, dimension=1536, metric='cosine')
                        vectores = Pinecone.from_documents(docsearch, embeddings, index_name=index_name)
                        print('Ok')
                        
                    vector_store=Pinecone.from_documents(texts,embeddings,index_name=index_n)
                    return "archivo cargado " + index_n                  
            
            except Exception as e:
                print("Error:", str(e))

    # async def generate_embeddings(self, file:UploadFile=File):
    #     try:
    #             #Leer archivo
    #             from langchain.document_loaders import PyPDFLoader
    #             with open(os.path.join(self.path, file.filename), "wb") as f:
    #                 content = await file.read()
    #                 f.write(content)
    #                 f.close()
    #                 data=f"./{file.filename}"
    #                 reader = PyPDFLoader(data)
    #                 fileload = reader.load()
                                       
    #                 #fragmentar los textos
    #                 text_splitter = RC(        
    #                     chunk_size = 1000,
    #                     chunk_overlap  = 50,
    #                     length_function = len,
    #                 )
    #                 docs = text_splitter.split_documents(fileload)
                    
    #                 print(len(docs))
    #                 print(docs[0])

    #                 docs = [str(i.page_content) for i in docs] #Lista de parrafos
    #                 parrafos = pd.DataFrame(docs, columns=["texto"])
    #                 parrafos['Embedding'] = parrafos["texto"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002')) # Nueva columna con los embeddings de los 
    #                 parrafos.to_csv('MTG.csv')
    #                 print(parrafos)

    #                 embeddings_list = []
    #                 for idx, row in parrafos.iterrows():
    #                     embedding = row["Embedding"]
    #                     embeddings_list.append({"text": row["texto"], "embedding": embedding})
                    

    #                 # embeddings = OpenAIEmbeddings()

    #                 nombre, extension = os.path.splitext(file.filename)
    #                 key=nombre.lower()

    #                 redis_client=RedisClient(os.environ.get('REDIS_HOST'), os.environ.get('REDIS_PORT'))   

    #                 if not redis_client.exists(key):         
    #                     time.sleep(1)
    #                     print("Creando embedding para: "+key)
    #                     redis_client.set(key, json.dumps(embeddings_list))  

    #                 else:
    #                     embeddings_data = redis_client.get(key)
    #                     embeddings_list = json.loads(embeddings_data)

    #                 return embeddings_list           
                  
            
    #     except Exception as e:
    #         print("Error:", str(e))
       