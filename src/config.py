"""Configuration settings for the condenser retrieval system."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv

# Load environment variables from .env file (optional)
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    """Configuration dataclass for condenser retrieval system."""

    DATA_DIR: Path = PROJECT_ROOT / "data"
    CORPUS_PATH: Path = PROJECT_ROOT / "data" / "corpus.jsonl"
    TRAIN_PAIRS_PATH: Path = PROJECT_ROOT / "data" / "train_pairs.tsv"
    VAL_PAIRS_PATH: Path = PROJECT_ROOT / "data" / "val_pairs.tsv"
    MODEL_NAME: str = "bert-base-uncased"
    OUTPUT_DIR: Path = PROJECT_ROOT / "checkpoints"
    BATCH_SIZE: int = 32
    MAX_LEN: int = 256
    LR: float = 2e-5
    EPOCHS: int = 2
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    FAISS_INDEX_DIR: Path = PROJECT_ROOT / "indexes"
    FAISS_INDEX_NAME: str = "faiss_flat.index"

    def __post_init__(self):
        """Post-initialization to read from environment variables and convert paths."""
        # Read from environment variables if present
        self.DATA_DIR = Path(os.getenv("DATA_DIR", str(self.DATA_DIR)))
        self.CORPUS_PATH = Path(os.getenv("CORPUS_PATH", str(self.CORPUS_PATH)))
        self.TRAIN_PAIRS_PATH = Path(os.getenv("TRAIN_PAIRS_PATH", str(self.TRAIN_PAIRS_PATH)))
        self.VAL_PAIRS_PATH = Path(os.getenv("VAL_PAIRS_PATH", str(self.VAL_PAIRS_PATH)))
        self.MODEL_NAME = os.getenv("MODEL_NAME", self.MODEL_NAME)
        self.OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(self.OUTPUT_DIR)))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", self.BATCH_SIZE))
        self.MAX_LEN = int(os.getenv("MAX_LEN", self.MAX_LEN))
        self.LR = float(os.getenv("LR", self.LR))
        self.EPOCHS = int(os.getenv("EPOCHS", self.EPOCHS))
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)
        self.FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", str(self.FAISS_INDEX_DIR)))
        self.FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", self.FAISS_INDEX_NAME)


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get singleton Config instance.

    Returns:
        Config: Singleton configuration instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config

