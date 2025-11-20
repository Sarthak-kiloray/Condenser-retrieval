from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.index.search_index import search_index

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    k: int

class SearchResult(BaseModel):
    doc_id: str
    text: str
    score: float

@app.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest):
    results = search_index(request.query, k=request.k)
    return results

@app.get("/health")
async def health():
    return {"status": "ok"}
