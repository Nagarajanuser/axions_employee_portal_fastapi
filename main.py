from turtle import st

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import requests
from pydantic import BaseModel
import logging

from embedder_sentence_transformer import embed_user_query
from vectorstore import search_pinecone
from llm_llama3 import query_llm_with_context

# -----------------------------------
# ALLOWED ORIGINS
# -----------------------------------
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200"
]

# -----------------------------------
# RESPONSE MODEL
# -----------------------------------
class ChatResponse(BaseModel):
    status: str
    question: str
    answer: str
    retrieved_chunks: int

# -----------------------------------
# REQUEST MODEL
# -----------------------------------
class ChatRequest(BaseModel):
    user_input: str

app = FastAPI()

# -----------------------------------
# ADD CORS MIDDLEWARE
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "FastAPI + Ollama Running"}

@app.post("/chat")
def chat(chat_request: ChatRequest):

    try:
        # Embed the user's query to create a vector representation
        query_embedding = embed_user_query(chat_request.user_input)
        print("Query embedding shape:", len(query_embedding))

        #Search for relevant chunks in Pinecone vector database
        relevant_chunks = search_pinecone(query_embedding, top_k=4, namespace="")
        print("Relevant chunks len:", len(relevant_chunks))

        #Generate answer using retrieved chunks and the user query
        answer = query_llm_with_context(chat_request.user_input, relevant_chunks)
        print("Answer:", answer)

        return ChatResponse(
            status="success",
            question=chat_request.user_input,
            answer=answer,
            retrieved_chunks=len(relevant_chunks)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )

