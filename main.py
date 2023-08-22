import io
import os

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile,File

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from models.model import Prompt

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
async def uploadfile(file: UploadFile = File(...)):
    pdf_buffer = io.BytesIO(await file.read())
    reader = PdfReader(pdf_buffer)
    file_path = os.path.join(os.environ["UPLOAD_DIR"], file.filename)
    with open(file_path, "wb") as f:
        f.write(pdf_buffer.getbuffer())
        

    return {"file_path": file_path}

@app.post("/juridica")  
async def generate_response_pdf(prompt:Prompt, file: UploadFile = File(...)):
   
    #Captura el prompt
    query = prompt.text
    #Carga el archivo
    pdf_buffer=uploadfile(file)
    print(pdf_buffer)
    



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
   

    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI

    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    print(reader)
  

    docs = docsearch.similarity_search(query)
    return chain.run(input_documents=docs, question=query)


"""
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
"""