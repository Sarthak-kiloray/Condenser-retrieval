"""Dataset loading utilities."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import Dataset, load_dataset


def load_retrieval_dataset(name: str = "squad", split: str = "train") -> Dataset:
    """Load a dataset for retrieval tasks.

    Args:
        name: Dataset name
        split: Dataset split

    Returns:
        Loaded dataset
    """
    dataset = load_dataset(name, split=split)
    return dataset


def prepare_contrastive_pairs(
    dataset: Dataset,
    query_field: str = "question",
    doc_field: str = "context",
) -> List[Tuple[str, str]]:
    """Prepare positive query-document pairs for contrastive learning.

    Args:
        dataset: Input dataset
        query_field: Field name for queries
        doc_field: Field name for documents

    Returns:
        List of (query, document) pairs
    """
    pairs = []
    for example in dataset:
        if query_field in example and doc_field in example:
            pairs.append((example[query_field], example[doc_field]))
    return pairs


def load_corpus(path: str) -> List[dict]:
    """Load corpus from JSONL file.

    Each line should be a JSON object with "id" and "text" fields.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries with "id" and "text" keys
    """
    corpus = []
    path_obj = Path(path)
    if not path_obj.exists():
        return corpus

    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if "id" in doc and "text" in doc:
                corpus.append({"id": doc["id"], "text": doc["text"]})
    return corpus


def load_pairs(path: str) -> List[Tuple[str, str]]:
    """Load query-document pairs from TSV file.

    Format: query\tpositive_doc_text

    Args:
        path: Path to TSV file

    Returns:
        List of (query, positive_doc_text) tuples
    """
    pairs = []
    path_obj = Path(path)
    if not path_obj.exists():
        return pairs

    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def load_triplets(path: str) -> Optional[List[Tuple[str, str, str]]]:
    """Load query-document triplets from TSV file (optional hard negatives).

    Format: query\tpositive_doc_text\tnegative_doc_text

    Args:
        path: Path to TSV file

    Returns:
        List of (query, pos_text, neg_text) tuples if file exists, None otherwise
    """
    triplets = []
    path_obj = Path(path)
    if not path_obj.exists():
        return None

    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                triplets.append((parts[0], parts[1], parts[2]))
    return triplets

