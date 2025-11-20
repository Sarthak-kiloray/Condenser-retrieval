"""Build FAISS index from document embeddings."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer

from src.config import get_config
from src.data.datasets import load_corpus
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

    print(f"Loading checkpoint from {checkpoint_path}")

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
        print(f"Tokenizer not found at {tokenizer_path}, using default")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def encode_texts_batch(
    texts: list,
    model: CondenserModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 256,
    batch_size: int = 32,
):
    """Encode a list of texts to embeddings in batches.

    Args:
        texts: List of text strings
        model: Condenser model
        tokenizer: Tokenizer instance
        device: Device to run inference on
        max_length: Maximum sequence length
        batch_size: Batch size for encoding

    Returns:
        Tensor of embeddings [num_texts, embedding_dim]
    """
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            encoded = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Encode
            embeddings = model(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def build_index(
    corpus_path: str,
    checkpoint_dir: str,
    out_dir: str = "indexes",
    index_name: str = "faiss_flat.index",
    mapping_name: str = "mapping.jsonl",
    batch_size: int = 32,
):
    """Build FAISS index from corpus documents.

    Args:
        corpus_path: Path to corpus JSONL file
        checkpoint_dir: Directory containing checkpoint and tokenizer
        out_dir: Output directory for index and mapping
        index_name: Name of the index file
        mapping_name: Name of the mapping file
        batch_size: Batch size for encoding
    """
    config = get_config()
    device = torch.device(config.DEVICE)

    # Load model and tokenizer
    checkpoint_dir_path = Path(checkpoint_dir)
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir_path, device)

    # Load corpus
    corpus = load_corpus(corpus_path)
    if not corpus:
        raise ValueError(f"No corpus found at {corpus_path}")
    print(f"Loaded {len(corpus)} documents from corpus")

    # Extract texts
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_ids = [doc["id"] for doc in corpus]

    # Encode corpus in batches
    print("Encoding corpus documents...")
    embeddings = encode_texts_batch(
        corpus_texts,
        model,
        tokenizer,
        device,
        max_length=config.MAX_LEN,
        batch_size=batch_size,
    )

    # Convert to numpy array (float32 for FAISS)
    embeddings_array = embeddings.numpy().astype("float32")
    dimension = embeddings_array.shape[1]
    print(f"Encoded {len(corpus)} documents to embeddings of dimension {dimension}")

    # Build FAISS index (IndexFlatIP for dot product)
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)  # Inner product for dot product similarity
    index.add(embeddings_array)
    print(f"Built index with {index.ntotal} vectors")

    # Create output directory
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Save index
    index_path = out_dir_path / index_name
    faiss.write_index(index, str(index_path))
    print(f"Saved index to {index_path}")

    # Save mapping file: vector position -> doc id/text
    mapping_path = out_dir_path / mapping_name
    with open(mapping_path, "w", encoding="utf-8") as f:
        for pos, doc in enumerate(corpus):
            mapping_entry = {
                "position": pos,
                "doc_id": doc["id"],
                "text": doc["text"],
            }
            f.write(json.dumps(mapping_entry, ensure_ascii=False) + "\n")
    print(f"Saved mapping to {mapping_path}")

    print(f"\nIndexing complete!")
    print(f"  Index: {index_path}")
    print(f"  Mapping: {mapping_path}")
    print(f"  Total vectors: {index.ntotal}")


if __name__ == "__main__":
    config = get_config()

    parser = argparse.ArgumentParser(description="Build FAISS index from corpus")
    parser.add_argument("--corpus", type=str, default=str(config.CORPUS_PATH))
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=str(config.FAISS_INDEX_DIR))
    parser.add_argument("--index_name", type=str, default=config.FAISS_INDEX_NAME)
    parser.add_argument("--mapping_name", type=str, default="mapping.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    build_index(
        corpus_path=args.corpus,
        checkpoint_dir=args.checkpoint_dir,
        out_dir=args.out_dir,
        index_name=args.index_name,
        mapping_name=args.mapping_name,
        batch_size=args.batch_size,
    )
