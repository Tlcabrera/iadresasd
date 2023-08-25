from fastapi import APIRouter, UploadFile,File
from uploadservice import UploadService
from models.model import Prompt
import numpy as np
import pinecone
import openai
import os


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

routes=APIRouter()


#endopoints

@routes.get("/")
def welcome():
    return "Bienvenido a IADRES"

@routes.post("/load-file")  
async def generate_response_pdf(prompt:str, file: UploadFile = File(...)):
    try:
        #Carga y embedding del archivo
        data=await UploadService().analize_file(file)
        print(data)
        #Responde si se envia el similarity search aquí dentro      

        chain = load_qa_chain(OpenAI(), chain_type="stuff")  
       
        docs = data.similarity_search(prompt)
        return chain.run(input_documents=docs, question=prompt)

    
    except Exception as e:
        print("Error:", str(e))

@routes.post("/load-data")
async def load_pdf(file: UploadFile = File(...)):
    data=await UploadService().embedding_text(file)

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
    # try:
    #     pinecone.init(api_key=os.environ.get('PINECONE_APY_KEY'), environment=os.environ.get('PINECONE_ENV'))

    #     # Crear una instancia del índice
    #     index = pinecone.Index(index_name.text)

    #     # Inicializar el modelo SentenceTransformer
    #     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
    #     # Texto de consulta
    #     prompt_text = prompt.text
        
    #     # Generar el vector numérico para la consulta
    #     query_vector = model.encode([prompt_text])[0]

    #     desired_dimension = 1536
    #     if query_vector.shape[0] != desired_dimension:
    #         query_vector = np.pad(query_vector, (0, desired_dimension - query_vector.shape[0]), mode='constant')

    #     # Convertir el ndarray a una lista
    #     query_vector_list = query_vector.tolist()
        
    #     # Realizar la búsqueda de similitud utilizando el método query()
    #     top_k = 1  # Número de resultados relevantes a retornar
    #     search_results = index.query(queries=[query_vector_list], top_k=top_k)

    #     print(search_results)

    #     return search_results
             
          
    # except Exception as e:
    #     return f"Error: {e}"
    

