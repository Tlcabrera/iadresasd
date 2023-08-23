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
                docsearch = FAISS.from_texts(texts, embeddings)
                print(f'Embeddings{docsearch}')

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

           

                

        except Exception as e:
            print("Error:", str(e))
