from pymongo import MongoClient
MONGO_HOST="localhost"
MONGO_PUERTO="27017"
MONGO_URI="mongodb://"+MONGO_HOST+":"+MONGO_PUERTO+"/"
try:
    conn = MongoClient(MONGO_URI)
    db = conn.promptsdb
    collection_prompt = db["prompt"]
    collection_embeddings = db["embeddings"]
except:
    print("Error de conexión")