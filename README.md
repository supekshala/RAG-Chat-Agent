# rag-chat
FastAPI app integrates OpenAI and Pinecone to handle PDF uploads and question-answering through retrieval-augmented generation (RAG). It processes PDFs, stores data in a vector store (Pinecone), and uses history-aware retrieval for Q&amp;A tasks.  Resources


# run backend

uvicorn main:app --reload --env-file .env
python3 -m http.server 8080
pip install -r requirements.txt


