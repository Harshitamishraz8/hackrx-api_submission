import os
import hashlib
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import uvicorn

from app.ingest import process_and_store_pdf
from app.query import query_documents
from app.llm import query_llm

# Load environment variables
load_dotenv()

app = FastAPI(title="HackRx Document Q&A API", version="1.0.0")

# Security
security = HTTPBearer()
HACKRX_API_TOKEN = os.getenv("HACKRX_API_TOKEN", "hackrx-secret-token")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKRX_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Request/Response models
class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

@app.get("/")
async def health_check():
    return {
        "status": "API is running",
        "message": "HackRx Document Q&A API",
        "endpoint": "/hackrx/run"
    }

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    try:
        print(f"Processing {len(request.questions)} questions for document: {request.documents}")
        
        # Generate document ID from URL
        doc_id = hashlib.md5(request.documents.encode()).hexdigest()[:8]
        
        # Process and store PDF in Pinecone
        success = await process_and_store_pdf(request.documents, doc_id)
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to process PDF document"
            )
        
        # Query for relevant contexts for each question
        all_contexts = []
        for question in request.questions:
            contexts = query_documents(question, doc_id, top_k=5)
            all_contexts.extend(contexts)
        
        # Remove duplicates while preserving order
        unique_contexts = []
        seen = set()
        for context in all_contexts:
            if context not in seen:
                unique_contexts.append(context)
                seen.add(context)
        
        # Generate answers using LLM
        answers = query_llm(request.questions, unique_contexts)
        
        print(f"Generated {len(answers)} answers")
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        print(f"Error in /hackrx/run: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)