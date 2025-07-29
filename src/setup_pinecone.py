import pinecone
from pinecone import Pinecone,ServerlessSpec

from loguru import logger
import os
from dotenv import load_dotenv
import yaml 

# configure logging
#logger.add("logs/chatbot.log",rotation="1 MB")

def intialize_pinecone():
    logger.info("Intializing pinecone index")

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    #evironment = os.getenv("PINECONE_ENVIROMENT")

    if not api_key :
        logger.error("PINECONE_API_KEY is not found in .env")
        raise ValueError("missing pinecone credentials")
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    index_name = config["pinecone"]["index_name"]
    dimensions = config["pinecone"]["dimension"]

   
    pc = Pinecone(api_key=api_key)

    if index_name in pc.list_indexes().names():
        logger.info("Deleting existing indexses")
        pc.delete_index(index_name)

    logger.info(f"Creating index in pincone with {index_name} and {dimensions} ")    
    try: 

         pc.create_index(
             
             name=index_name,
             dimension=dimensions,
             metric="cosine",
             spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
  
    except Exception as e:
        logger.error("failed creating pinecone index")
        raise

    logger.info(f"Index {index_name} created succesfully")
    

if __name__ == "__main__":
    intialize_pinecone()