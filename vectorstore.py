from pinecone import Pinecone
import os
from typing import List
from dotenv import load_dotenv

#Load environment variables from .env file
load_dotenv()

#Initialize Pinecone client
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  
index = pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME"))

def store_in_pinecone(chunks: List[str], embeddings: List[List[float]], namespace: str = ""):
    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_data = {
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "chunk_index": i}
        }
        #print(f"Prepared vector for chunk {i}")
        vectors_to_upsert.append(vector_data)
    #Upsert vectors in batches (Pinecone recommends batch sizes of 100)
    batch_size = 100
    #print(f"batch_size: {batch_size}")
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        #print(f"Upserting batch {i}")
        index.upsert(vectors=batch, namespace=namespace)

# Search chunk from the Pinecone vector database using the query embedding
def search_pinecone(query_embedding: List[float], top_k: int = 4, namespace: str = ""):
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    print(f"Search results: {len(search_results.matches)}")
    relevant_chunks = []
    for match in search_results.matches:
        relevant_chunks.append(match.metadata.get("text", ""))
    return relevant_chunks