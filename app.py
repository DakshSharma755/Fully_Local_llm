from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from contextlib import asynccontextmanager
import rag_core
import shutil

class QueryRequest(BaseModel):
    query: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO:     Application startup...")
    rag_core.initialize() 
    yield
    print("INFO:     Application shutdown...")

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    """
    A simple health-check endpoint to confirm the server is running.
    """
    return {"status": "RAG Chatbot API is running"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    The main endpoint to handle user queries.
    It takes a JSON with a 'query' field and returns
    the generated answer.
    """
    print(f"INFO:     Received query: {request.query}")

    answer = rag_core.get_answer(request.query)
    
    return {"answer": answer}

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    """
    Receives a PDF file, saves it, and triggers the indexing process.
    """
    save_path = Path(rag_core.SOURCE_DOCS_DIR) / file.filename
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(f"INFO:     Received and saved file: {file.filename}")
    
    success = rag_core.process_and_add_document(save_path)
    
    if success:
        return {"status": "success", "filename": file.filename, "message": "File processed and added to the knowledge base."}
    else:
        return {"status": "error", "filename": file.filename, "message": "Failed to process the file."}