import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pinecone import Pinecone

# We import rag here. Note that model loading will happen lazily on the first request 
# or can be forced during startup.
from rag import get_answer
from scraper import ingest_data, PINECONE_API_KEY, PINECONE_INDEX_NAME

app = FastAPI(title="OneWorld Technologies AI Chatbot")

# CORS middleware so the frontend can reach the API regardless of origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pathlib
# Serve frontend static files
frontend_dir = pathlib.Path(__file__).parent.parent / "frontend"
app.mount("/assets", StaticFiles(directory=str(frontend_dir)), name="assets")

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    print("Starting background loading of model and checking vector database...")
    import threading

    def bootstrap():
        try:
            if PINECONE_API_KEY and PINECONE_API_KEY != "your_pinecone_api_key_here":
                pc = Pinecone(api_key=PINECONE_API_KEY)
                if PINECONE_INDEX_NAME in pc.list_indexes().names():
                    current_count = pc.Index(PINECONE_INDEX_NAME).describe_index_stats().get("total_vector_count", 0)
                    print(f"Pinecone index '{PINECONE_INDEX_NAME}' currently has {current_count} vector(s).")
                    if current_count == 0:
                        print("Index is empty. Running initial ingestion...")
                        ingest_data()
                else:
                    print(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Running initial ingestion...")
                    ingest_data()
            else:
                print("Skipping auto-ingest because PINECONE_API_KEY is missing or invalid.")
        except Exception as e:
            print(f"Startup ingestion check failed: {e}")

    threading.Thread(target=bootstrap, daemon=True).start()

@app.get("/")
async def root():
    return FileResponse("../frontend/index.html")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.message:
        return JSONResponse(status_code=400, content={"error": "Message cannot be empty."})
    print(f"\n[USER PROMPT] {request.message}")
    
    from rag import get_answer_stream
    
    def generate():
        for chunk in get_answer_stream(request.message):
            # Print to backend console as it streams
            print(chunk, end="", flush=True)
            yield chunk
        print("\n")

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/api/ingest")
async def ingest():
    """Trigger the scraper and Pinecone ingestion process."""
    try:
        # Run blocking ingestion in a separate thread
        await asyncio.to_thread(ingest_data)
        return {"status": "Success. Data ingested to Pinecone."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
