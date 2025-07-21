# -*- coding: utf-8 -*-
"""embeddings.ipynb



Original file is located at
    https://colab.research.google.com/drive/1AdaF4BDBwmD3PufDYjZy-hLSsQUnxdKP
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import psutil
import os

def generate_embeddings(input_path,output_dir,batch_size=10):
  with open(input_path,'r') as f:
    data = json.load(f)

  model = SentenceTransformer('all-MiniLM-L6-v2')

  os.makedirs(output_dir,exist_ok=True)

  embeddings = []
  batch = []
  batch_ids = []
  batch_metadata = []

  for item in data:
    batch.append(item['text'])
    batch_ids.append(item['id'])
    batch_metadata.append(item['metadata'])

    if len(batch) >= batch_size:
      memory = psutil.virtual_memory()
      print(f"Memory usage: {memory.used / 1024 / 1024 / 1024:.2f} GB")
      batch_embeddings = model.encode(batch,batch_size=batch_size)
      for i,emb in enumerate(batch_embeddings):
        embeddings.append({
            'id':batch_ids[i],
            'embedding':emb.tolist(),
            'metadata':batch_metadata[i]
        })
      batch = []
      batch_ids = []
      batch_metadata = []



  if batch:
    memory = psutil.virtual_memory()
    batch_embeddings = model.encode(batch,batch_size=batch_size)
    for i,emb in enumerate(batch_embeddings):
      embeddings.append({
          'id':batch_ids[i],
          'embedding':emb.tolist(),
          'metadata':batch_metadata[i]
      })

  output_path = os.path.join(output_dir,"embedding.json")
  with open(output_path,"w")  as f:
    json.dump(embeddings,f,indent=2)









if __name__ == "__main__":
  input_path = "/content/faqs_processed.json"
  output_dir = "/content/embeddings"
  batch_size = 10
  generate_embeddings(input_path,output_dir,batch_size)