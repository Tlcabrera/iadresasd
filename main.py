
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from models.model import Prompt

# Get your API keys from openai, you will need to create an account. 
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview


app=FastAPI()
load_dotenv()

import os
api_key = os.environ["OPENAI_API_KEY"]

# connect your Google Drive
reader = PdfReader('./E54051222045753R001297402700.pdf')
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
docsearch

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

"""
query = "cual es el nombre completo del paciente"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents=docs, question=query))
"""

#endopoints
@app.post("/juridica")  
def generate_response_pdf(prompt:Prompt):
   query = prompt.text
   docs = docsearch.similarity_search(query)
   return chain.run(input_documents=docs, question=query)


