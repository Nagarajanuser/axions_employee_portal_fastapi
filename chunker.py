from typing import List, Tuple

def chunk_pages(pages: List[str], chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """
    Chunk the pages into smaller pieces of specified size.
    
    Args:
        pages (List[str]): List of page texts.
        chunk_size (int): Maximum number of characters in each chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    chunks: List[str] = []

    full_text = " ".join(pages)
    text_length = len(full_text)

    if text_length == 0:
        return chunks
    
    start = 0
    while start < text_length:
        # Calculate the end index for the current chunk
        end = min(start + chunk_size, text_length)

        # Extract the chunk of text
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end >= text_length:
            break
        
        #Calculate the start index for the next chunk with overlap
        start += chunk_size - chunk_overlap
        #print(f"Chunk created from index {start} to {end}. Chunk length: {len(chunk)} characters.")

    return chunks