"""Data preprocessing utilities."""

import json
from pathlib import Path
from typing import List, Optional

from transformers import AutoTokenizer

from src.config import get_config


def tokenize_texts(
    texts: List[str],
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 512,
    padding: bool = True,
    truncation: bool = True,
) -> dict:
    """Tokenize a list of texts.

    Args:
        texts: List of text strings
        tokenizer_name: Name of the tokenizer
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences

    Returns:
        Tokenized inputs
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )


def batch_tokenize(
    texts: List[str],
    batch_size: int = 32,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 512,
) -> List[dict]:
    """Tokenize texts in batches.

    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        tokenizer_name: Name of the tokenizer
        max_length: Maximum sequence length

    Returns:
        List of tokenized batches
    """
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batches.append(tokenize_texts(batch, tokenizer_name, max_length))
    return batches


def clean(text: str) -> str:
    """Basic text normalization.

    Args:
        text: Input text string

    Returns:
        Normalized text (lowercased and stripped)
    """
    return text.lower().strip()


def build_demo_corpus() -> None:
    """Create a tiny fallback corpus and persist to data/corpus.jsonl.

    Creates 10-20 synthetic documents for quick smoke testing.
    """
    config = get_config()
    corpus_path = config.CORPUS_PATH
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    demo_docs = [
        {
            "id": "doc_001",
            "text": "Python is a high-level programming language known for its simplicity and readability. It is widely used in data science, web development, and artificial intelligence.",
        },
        {
            "id": "doc_002",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. It powers many modern applications.",
        },
        {
            "id": "doc_003",
            "text": "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand and generate text.",
        },
        {
            "id": "doc_004",
            "text": "Deep learning uses neural networks with multiple layers to learn complex patterns in data. It has revolutionized image recognition, speech processing, and language understanding.",
        },
        {
            "id": "doc_005",
            "text": "Information retrieval is the process of finding relevant documents from a large collection. It is fundamental to search engines and recommendation systems.",
        },
        {
            "id": "doc_006",
            "text": "Vector embeddings are dense representations of text or objects in a high-dimensional space. They capture semantic meaning and enable similarity search.",
        },
        {
            "id": "doc_007",
            "text": "Contrastive learning is a training technique that learns representations by pulling similar items together and pushing dissimilar ones apart in embedding space.",
        },
        {
            "id": "doc_008",
            "text": "FAISS is a library for efficient similarity search and clustering of dense vectors. It can handle billions of vectors and is widely used in production systems.",
        },
        {
            "id": "doc_009",
            "text": "Transformers are neural network architectures that use attention mechanisms to process sequences. They have become the foundation of modern NLP models.",
        },
        {
            "id": "doc_010",
            "text": "BERT is a bidirectional transformer model that learns contextualized word representations. It has been pre-trained on large text corpora and can be fine-tuned for various tasks.",
        },
        {
            "id": "doc_011",
            "text": "Dense retrieval uses learned embeddings to find relevant documents. It is more efficient than traditional keyword-based search and can capture semantic meaning.",
        },
        {
            "id": "doc_012",
            "text": "Semantic search goes beyond keyword matching to understand the meaning and intent behind queries. It uses AI to find contextually relevant results.",
        },
        {
            "id": "doc_013",
            "text": "Fine-tuning adapts pre-trained models to specific tasks by training on domain-specific data. It requires less data and computation than training from scratch.",
        },
        {
            "id": "doc_014",
            "text": "Tokenization is the process of breaking text into smaller units called tokens. It is the first step in most NLP pipelines and affects model performance.",
        },
        {
            "id": "doc_015",
            "text": "Query understanding is the ability to interpret user queries and extract intent. It is crucial for building effective search and retrieval systems.",
        },
    ]

    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in demo_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Created demo corpus with {len(demo_docs)} documents at {corpus_path}")

