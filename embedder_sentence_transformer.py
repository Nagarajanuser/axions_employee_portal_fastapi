from sentence_transformers import SentenceTransformer
from typing import List

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    
    # Remove empty chunks
    clean_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Generate embeddings in batch
    embeddings = model.encode(
        clean_chunks,
        convert_to_tensor=False,   # returns numpy array
        show_progress_bar=True
    )

    return embeddings.tolist()

def embed_user_query(query: str) -> List[float]:
    """Embed the user's query into a vector representation."""
    query_embedding = model.encode(
        query,
        convert_to_tensor=False
    )
    return query_embedding.tolist()