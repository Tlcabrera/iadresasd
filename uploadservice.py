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
import pinecone 

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter as RC
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone, FAISS
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
                    index_name=nombre.lower()

                    if index_name not in pinecone.list_indexes():
                        print(f"Creando el índice {index_name} ...")
                        pinecone.create_index(index_name,dimension=1536,metric='cosine')
                        print("Done!")
                    else:
                        print(f"El índice {index_name} ya existe")

                    vector_store=Pinecone.from_documents(texts,embeddings,index_name=index_name)
                                       

            except Exception as e:
                print("Error:", str(e))
