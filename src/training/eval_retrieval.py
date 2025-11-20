"""Evaluation script for retrieval tasks."""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.config import get_config
from src.data.datasets import load_corpus, load_pairs
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


def encode_texts(
    texts: List[str],
    model: CondenserModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 256,
    batch_size: int = 32,
):
    """Encode a list of texts to embeddings.

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


def compute_similarity(query_emb: torch.Tensor, doc_embs: torch.Tensor, use_cosine: bool = True):
    """Compute similarity between query and documents.

    Args:
        query_emb: Query embedding [dim]
        doc_embs: Document embeddings [num_docs, dim]
        use_cosine: If True, use cosine similarity; else use dot product

    Returns:
        Similarity scores [num_docs]
    """
    if use_cosine:
        # Cosine similarity
        query_emb = F.normalize(query_emb, p=2, dim=0)
        doc_embs = F.normalize(doc_embs, p=2, dim=1)
        scores = torch.matmul(doc_embs, query_emb)
    else:
        # Dot product
        scores = torch.matmul(doc_embs, query_emb)

    return scores


def evaluate_retrieval(
    corpus_path: str,
    queries_path: str,
    checkpoint_dir: str,
    k: int = 10,
    use_cosine: bool = True,
    ground_truth_pairs_path: Optional[str] = None,
):
    """Evaluate retrieval performance.

    Args:
        corpus_path: Path to corpus JSONL file
        queries_path: Path to queries (TSV file with queries, can extract just queries)
        checkpoint_dir: Directory containing checkpoint and tokenizer
        k: Number of top results to retrieve
        use_cosine: If True, use cosine similarity; else use dot product
        ground_truth_pairs_path: Optional path to ground truth pairs (TSV) for Recall@k
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

    # Load queries (extract queries from pairs file)
    pairs = load_pairs(queries_path)
    queries = [pair[0] for pair in pairs]  # Extract just queries
    if not queries:
        raise ValueError(f"No queries found at {queries_path}")
    print(f"Loaded {len(queries)} queries")

    # Load ground truth if available
    ground_truth = None
    if ground_truth_pairs_path:
        gt_pairs = load_pairs(ground_truth_pairs_path)
        # Build ground truth: for each query, find which corpus doc IDs match
        ground_truth = []
        corpus_texts = [doc["text"] for doc in corpus]
        for query, pos_text in gt_pairs:
            # Find matching corpus documents
            matching_indices = []
            for idx, doc in enumerate(corpus):
                if pos_text.lower() in doc["text"].lower() or doc["text"].lower() in pos_text.lower():
                    matching_indices.append(idx)
            ground_truth.append(matching_indices)
        print(f"Loaded ground truth for {len(ground_truth)} queries")

    # Encode corpus
    print("Encoding corpus...")
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_embeddings = encode_texts(
        corpus_texts,
        model,
        tokenizer,
        device,
        max_length=config.MAX_LEN,
    )
    print(f"Encoded corpus to {corpus_embeddings.shape}")

    # Encode queries
    print("Encoding queries...")
    query_embeddings = encode_texts(
        queries,
        model,
        tokenizer,
        device,
        max_length=config.MAX_LEN,
    )
    print(f"Encoded queries to {query_embeddings.shape}")

    # Retrieve top-k for each query
    print(f"\nRetrieving top-{k} results for each query...")
    all_recalls = []

    for i, query in enumerate(queries):
        query_emb = query_embeddings[i]
        scores = compute_similarity(query_emb, corpus_embeddings, use_cosine=use_cosine)

        # Get top-k indices
        top_k_indices = torch.topk(scores, k=min(k, len(corpus)), dim=0).indices.tolist()
        top_k_scores = scores[top_k_indices].tolist()

        print(f"\nQuery {i+1}: {query}")
        print(f"Top-{k} retrieved documents:")

        for rank, (idx, score) in enumerate(zip(top_k_indices, top_k_scores), 1):
            doc = corpus[idx]
            doc_text_preview = doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
            print(f"  [{rank}] (doc_id: {doc['id']}, score: {score:.4f})")
            print(f"      {doc_text_preview}")

        # Compute Recall@k if ground truth available
        if ground_truth and i < len(ground_truth):
            relevant_docs = set(ground_truth[i])
            retrieved_docs = set(top_k_indices)
            recall_k = len(relevant_docs & retrieved_docs) / len(relevant_docs) if relevant_docs else 0.0
            all_recalls.append(recall_k)
            print(f"  Recall@{k}: {recall_k:.4f} ({len(relevant_docs & retrieved_docs)}/{len(relevant_docs)} relevant retrieved)")

    # Report overall metrics
    if all_recalls:
        avg_recall = sum(all_recalls) / len(all_recalls)
        print(f"\n{'='*60}")
        print(f"Overall Recall@{k}: {avg_recall:.4f}")
        print(f"{'='*60}")


def main():
    """Main evaluation function."""
    config = get_config()

    parser = argparse.ArgumentParser(description="Evaluate retrieval model")
    parser.add_argument("--corpus", type=str, default=str(config.CORPUS_PATH))
    parser.add_argument("--queries", type=str, default=str(config.TRAIN_PAIRS_PATH))
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--use_dot", action="store_true", help="Use dot product instead of cosine similarity")
    parser.add_argument("--ground_truth", type=str, default=None, help="Path to ground truth pairs (TSV)")
    args = parser.parse_args()

    evaluate_retrieval(
        corpus_path=args.corpus,
        queries_path=args.queries,
        checkpoint_dir=args.checkpoint_dir,
        k=args.k,
        use_cosine=not args.use_dot,
        ground_truth_pairs_path=args.ground_truth,
    )


if __name__ == "__main__":
    main()
