import os

from dotenv import load_dotenv
from fastapi import FastAPI
from routes.routes import routes

#crear objeto de tipo API
app=FastAPI()
#cargar archivo de variables de entorno
load_dotenv()
#include de las rutas con el endpoint del proyecto
app.include_router(routes)
#Definir API_KEY de OPENAI
api_key = os.environ["OPENAI_API_KEY"]



    
    

       
    