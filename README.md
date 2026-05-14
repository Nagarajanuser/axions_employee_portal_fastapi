
Step: 1
    python -m venv venv

if above one fail, First delete "venv" folder after use below command
    python -m pip install --upgrade pip setuptools wheel
after again use this below command
    python -m venv venv

Step: 2
venv\Scripts\activate

Step: 3
pip install -r requirements.txt

-----------------------------------------------------------------------------
Required packages
-----------------------------------------------------------------------------
# Initial Required

pip install fastapi
pip install uvicorn
pip install requests

pip install pypdf
pip install dotenv

# to embedd
pip install sentence-transformers

# Upload the chunk to Vector database (pinecone)
pip install pinecone

# to get final aswer use LLM (user query + chunk data) 
  locall llm (no need to install)


-----------------------------------------------------------------------------
Use this APP
-----------------------------------------------------------------------------
To upload Documents
    python upload_documents.py

To run the APP
    uvicorn main:app --reload

To test ollama 
    http://localhost:11434/

To see in browser
    http://127.0.0.1:8000/

open api in swagger URL
    http://localhost:8000/docs
    
