# python imports
import os
import re
from os import getcwd

# FastAPI imports
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile,File

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from models.model import Prompt


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
                print(reader)
                # read data from the file and put them into a variable called raw_text
                raw_text = ''
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        raw_text += text

                print(raw_text[:100])
                # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

                text_splitter = CharacterTextSplitter(        
                    separator = "\n",
                    chunk_size = 1000,
                    chunk_overlap  = 200,
                    length_function = len,
                )
                texts = text_splitter.split_text(raw_text)

                print(len(texts))
                embeddings = OpenAIEmbeddings()
                docsearch = FAISS.from_texts(texts, embeddings)
                
                return docsearch
        except Exception as e:
            print("Error:", str(e))
