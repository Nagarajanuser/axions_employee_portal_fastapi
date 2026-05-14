
from pdfreader import read_pdf
from chunker import chunk_pages

from embedder_sentence_transformer import embed_chunks
from vectorstore import store_in_pinecone
from typing import List


pdf_path = "./resources/hr_axions.pdf"

def run():
    #Read HR Policy PDF and extract text
    pages = read_pdf(pdf_path)
    #print(f"Extracted {len(pages)} pages from the PDF.")
    #print(pages[3])

    #Chunk the data into smaller pieces 
    chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=150)
    #print(chunks[0])

    embeddings = embed_chunks(chunks)
    #print('embeddings   shape: ', len(embeddings), len(embeddings[0]))
    #print('embeddings   shape0: ', embeddings[0])

    #Store the chunks and their embeddings in Pinecone vector database
    store_in_pinecone(chunks, embeddings, namespace="")

if __name__ == "__main__":
    run()
