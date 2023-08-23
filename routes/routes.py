from fastapi import APIRouter, UploadFile,File
from uploadservice import UploadService

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

routes=APIRouter()


#endopoints

@routes.get("/juridica")
def welcome():
    return "Bienvenido a IADRES"

@routes.post("/juridica")  
async def generate_response_pdf(prompt:str, file: UploadFile = File(...)):
    try:
        #Carga y embedding del archivo
        data=await UploadService().analize_file(file)
        #Captura el prompt
        query=prompt
         
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI

        chain = load_qa_chain(OpenAI(), chain_type="stuff")        
        docs = data.similarity_search(query)
        return chain.run(input_documents=docs, question=query)
    
    except Exception as e:
        print("Error:", str(e))

    
