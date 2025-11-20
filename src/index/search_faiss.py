"""Search FAISS index."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer

from src.config import get_config
from src.models.condenser import CondenserModel


def load_model_and_tokenizer(checkpoint_dir: Path, device: torch.device):
    """Load model checkpoint and tokenizer.

    Args:
        checkpoint_dir: Directory containing checkpoint and tokenizer
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Find the latest checkpoint
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")

    # Sort by epoch number and get the latest
    checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
    checkpoint_path = checkpoint_files[-1]

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model name from checkpoint or use default
    model_name = checkpoint.get("model_name", "bert-base-uncased")

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
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def load_mapping(mapping_path: Path) -> List[dict]:
    """Load mapping file.

    Args:
        mapping_path: Path to mapping JSONL file

    Returns:
        List of mapping dictionaries
    """
    mapping = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mapping.append(json.loads(line))
    return mapping


def search_faiss(
    query: str,
    index_dir: str,
    checkpoint_dir: str,
    k: int = 5,
):
    """Search FAISS index for similar documents.

    Args:
        query: Query string
        index_dir: Directory containing index and mapping files
        checkpoint_dir: Directory containing checkpoint and tokenizer
        k: Number of top results to return

    Returns:
        List of matched documents with scores
    """
    config = get_config()
    device = torch.device(config.DEVICE)

    index_dir_path = Path(index_dir)
    checkpoint_dir_path = Path(checkpoint_dir)

    # Load index
    index_files = list(index_dir_path.glob("*.index"))
    if not index_files:
        # Try default name
        index_path = index_dir_path / "faiss_flat.index"
        if not index_path.exists():
            raise ValueError(f"No index file found in {index_dir}")
    else:
        index_path = index_files[0]

    print(f"Loading index from {index_path}")
    index = faiss.read_index(str(index_path))

    # Load mapping
    mapping_path = index_dir_path / "mapping.jsonl"
    if not mapping_path.exists():
        raise ValueError(f"Mapping file not found at {mapping_path}")
    mapping = load_mapping(mapping_path)
    print(f"Loaded mapping with {len(mapping)} entries")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir_path, device)

    # Encode query
    print(f"Encoding query: {query}")
    with torch.no_grad():
        encoded = tokenizer(
            query,
            max_length=config.MAX_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        query_embedding = model(input_ids, attention_mask)
        query_vector = query_embedding.cpu().numpy().astype("float32")

    # Search
    print(f"Searching for top-{k} results...")
    k = min(k, index.ntotal)
    distances, indices = index.search(query_vector, k)

    # Map results to documents
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        pos = int(idx)
        if pos < len(mapping):
            doc_info = mapping[pos]
            results.append(
                {
                    "position": pos,
                    "doc_id": doc_info["doc_id"],
                    "text": doc_info["text"],
                    "score": float(dist),
                }
            )

    return results


def main():
    """Main search function."""
    config = get_config()

    parser = argparse.ArgumentParser(description="Search FAISS index")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--index_dir", type=str, default=str(config.FAISS_INDEX_DIR))
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=5, help="Number of top results")
    args = parser.parse_args()

    results = search_faiss(
        query=args.query,
        index_dir=args.index_dir,
        checkpoint_dir=args.checkpoint_dir,
        k=args.k,
    )

    print(f"\n{'='*60}")
    print(f"Top-{len(results)} results for query: '{args.query}'")
    print(f"{'='*60}\n")

    for i, result in enumerate(results, 1):
        print(f"[{i}] Doc ID: {result['doc_id']}")
        print(f"    Score: {result['score']:.4f}")
        text_preview = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
        print(f"    Text: {text_preview}")
        print()


if __name__ == "__main__":
    main()
