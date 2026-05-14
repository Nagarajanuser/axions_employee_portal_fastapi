                
import requests

# -----------------------------
# OLLAMA CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:1b"


def query_llm_with_context(query: str, context: str):

    print("Query to Ollama:", query)
    print("Context to Ollama:", context)

    # 1. Define the System Instruction (Equivalent to OpenAI's system role)
    system_instruction = """
    You are a helpful assistant for answering user queries based on provided context. 
    Use the context to provide accurate and relevant answers. 
    Do not make assumptions beyond the context provided. 
    Let the user know if you cannot provide an answer based on the given context.
    """
    # -----------------------------
    # BUILD RAG PROMPT
    # -----------------------------
    final_prompt = f"""
    Context:
    {context}
    User Question:
    {query}
    Answer:
    """

    # -----------------------------
    # SEND TO OLLAMA
    # -----------------------------
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_instruction
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        answer = response.json()
        return answer['message']['content']
    else:
        return f"Error: {response.status_code}"
