import pinecone 
from pinecone import Pinecone
import json
from loguru import logger
import yaml
from dotenv import load_dotenv
import os

#logger.add("logs/chatbot.log",rotation="1 MB")

def upload_embeddings(embeddings_path,batch_size=10):
    logger.info("Starting embeddings upload to pinecone")

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    pc = Pinecone(api_key=api_key)    
    index = pc.Index(host="https://customer-support-2-ojqirsh.svc.aped-4627-b74a.pinecone.io")

    with open(embeddings_path,"r") as f:
        embeddings = json.load(f)

    if not embeddings:
        logger.error("Embedding json is empty")
        return

    logger.info(f"Uploading {len(embeddings)} embeddings in batches of {batch_size}")   
    for i in range(0,len(embeddings),batch_size):
        batch = embeddings[i:i + batch_size]
        vectors = [(item["id"],item["values"],item["metadata"]) for item in batch]
        try:
            index.upsert(vectors=vectors)
            logger.info(f"uploaded batch {batch_size} to pinecone")
        except Exception as e:
            logger.error("Failed to upload the batch")    

    logger.info("Embedding uploaded completed..")        

if __name__ == "__main__":
    embeddings_path = "data/embedded_data.json"
    upload_embeddings(embeddings_path,batch_size=10)



