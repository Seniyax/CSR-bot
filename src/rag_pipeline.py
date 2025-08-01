# -*- coding: utf-8 -*-
"""RAG_pipeline.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oAb0ymgMbucyYvXR1xWHxM6PxiHP510b
"""

import os
import getpass
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr


def load_template():
  path ="/content/prompt_template.txt"
  with open(path) as f:
    prompt_template = f.read()
  return PromptTemplate.from_template(prompt_template)


def rag_pipeline():
  load_dotenv("/content/.env")
  api_key = os.getenv("PINECONE_API_KEY")

  model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

  pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    max_new_tokens=150,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.1
   )

  llm = HuggingFacePipeline(pipeline=pipe)

  embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

  pc = Pinecone(api_key=api_key)
  index = pc.Index(host="https://customer-support-2-ojqirsh.svc.aped-4627-b74a.pinecone.io")
  vectorestore = PineconeVectorStore(index=index,embedding=embedding,text_key="text")

  top_k = 5

  prompt_template = load_template()

  rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorestore.as_retriever(),
    chain_type_kwargs={"prompt":prompt_template},
    return_source_documents=False
  )

  return rag_chain




  #query = "How do I fix my iPhone not charging?"
  #result = rag_chain.run(query)
  #print("Answer:", result)


rag_chain = rag_pipeline()

def ask_question(message,history):
    return rag_chain.run(message)

import gradio as gr

# Gradio UI
chatbot = gr.ChatInterface(
    fn=ask_question,
    title="🤖 RAG Chatbot",
    description=" Powered by Hugging Face's SmolLM2!",
    theme="soft"

)

chatbot.launch(share=True)