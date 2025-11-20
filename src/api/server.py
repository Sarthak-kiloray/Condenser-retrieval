"""FastAPI server for retrieval API."""

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer

from src.api.schemas import HealthResponse, QueryRequest, SearchResult
from src.config import get_config
from src.models.condenser import CondenserModel


app = FastAPI(title="Condenser Retrieval API", version="0.1.0")

# CORS middleware (allow localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model: CondenserModel = None
tokenizer: AutoTokenizer = None
index: faiss.Index = None
mapping: List[dict] = None
device: torch.device = None


def load_mapping(mapping_path: Path) -> List[dict]:
    """Load mapping file.

    Args:
        mapping_path: Path to mapping JSONL file

    Returns:
        List of mapping dictionaries
    """
    mapping_list = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mapping_list.append(json.loads(line))
    return mapping_list


@app.on_event("startup")
async def startup_event():
    """Initialize model, tokenizer, FAISS index and mapping on startup."""
    global model, tokenizer, index, mapping, device

    config = get_config()
    device = torch.device(config.DEVICE)

    # Find latest checkpoint directory
    checkpoint_files = list(Path(config.OUTPUT_DIR).glob("*/checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        raise RuntimeError(f"No checkpoint found in {config.OUTPUT_DIR}")

    checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
    checkpoint_path = checkpoint_files[-1]
    checkpoint_dir = checkpoint_path.parent

    print(f"Loading model from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", config.MODEL_NAME)

    # Load model
    model = CondenserModel(model_name=model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = checkpoint_dir / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        print(f"Tokenizer not found at {tokenizer_path}, using default")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load FAISS index
    index_files = list(config.FAISS_INDEX_DIR.glob("*.index"))
    if not index_files:
        index_path = config.FAISS_INDEX_DIR / config.FAISS_INDEX_NAME
        if not index_path.exists():
            raise RuntimeError(f"FAISS index not found in {config.FAISS_INDEX_DIR}")
    else:
        index_path = index_files[0]

    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))

    # Load mapping
    mapping_path = config.FAISS_INDEX_DIR / "mapping.jsonl"
    if not mapping_path.exists():
        raise RuntimeError(f"Mapping file not found at {mapping_path}")
    mapping = load_mapping(mapping_path)

    print(f"API initialized successfully!")
    print(f"  Model: {model_name}")
    print(f"  Index vectors: {index.ntotal}")
    print(f"  Mapping entries: {len(mapping)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/search", response_model=List[SearchResult])
async def search(request: QueryRequest):
    """Search endpoint for retrieval.

    Args:
        request: Query request with query text and k

    Returns:
        List of search results
    """
    if model is None or tokenizer is None or index is None or mapping is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Encode query
        with torch.no_grad():
            encoded = tokenizer(
                request.query,
                max_length=get_config().MAX_LEN,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            query_embedding = model(input_ids, attention_mask)
            query_vector = query_embedding.cpu().numpy().astype("float32")

        # Search FAISS
        k = min(request.k, index.ntotal)
        distances, indices = index.search(query_vector, k)

        # Map results to documents
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            pos = int(idx)
            if pos < len(mapping):
                doc_info = mapping[pos]
                results.append(
                    SearchResult(
                        doc_id=doc_info["doc_id"],
                        text=doc_info["text"],
                        score=float(dist),
                    )
                )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
