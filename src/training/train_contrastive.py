"""Training script for contrastive learning."""

import argparse
import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import get_config
from src.data.datasets import load_pairs, load_triplets
from src.models.condenser import CondenserModel
from src.models.losses import InfoNCE, TripletMarginLoss


class PairsDataset(Dataset):
    """Dataset for query-positive document pairs."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        """Initialize pairs dataset.

        Args:
            pairs: List of (query, positive_doc_text) tuples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """Get item by index.

        Args:
            idx: Item index

        Returns:
            Dictionary with tokenized query and document
        """
        query, doc = self.pairs[idx]

        query_encoded = self.tokenizer(
            query,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        doc_encoded = self.tokenizer(
            doc,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_encoded["input_ids"].squeeze(0),
            "query_attention_mask": query_encoded["attention_mask"].squeeze(0),
            "doc_input_ids": doc_encoded["input_ids"].squeeze(0),
            "doc_attention_mask": doc_encoded["attention_mask"].squeeze(0),
        }


class TripletsDataset(Dataset):
    """Dataset for query-positive-negative triplets."""

    def __init__(
        self,
        triplets: List[Tuple[str, str, str]],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        """Initialize triplets dataset.

        Args:
            triplets: List of (query, pos_text, neg_text) tuples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict:
        """Get item by index.

        Args:
            idx: Item index

        Returns:
            Dictionary with tokenized query, positive, and negative
        """
        query, pos_doc, neg_doc = self.triplets[idx]

        query_encoded = self.tokenizer(
            query,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        pos_encoded = self.tokenizer(
            pos_doc,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        neg_encoded = self.tokenizer(
            neg_doc,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_encoded["input_ids"].squeeze(0),
            "query_attention_mask": query_encoded["attention_mask"].squeeze(0),
            "pos_input_ids": pos_encoded["input_ids"].squeeze(0),
            "pos_attention_mask": pos_encoded["attention_mask"].squeeze(0),
            "neg_input_ids": neg_encoded["input_ids"].squeeze(0),
            "neg_attention_mask": neg_encoded["attention_mask"].squeeze(0),
        }


def train(
    train_pairs_path: str,
    val_pairs_path: Optional[str] = None,
    model_name: str = "bert-base-uncased",
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 32,
    out_dir: str = "checkpoints",
    max_length: int = 256,
    use_triplets: bool = False,
    triplets_path: Optional[str] = None,
):
    """Train condenser model with contrastive learning.

    Args:
        train_pairs_path: Path to training pairs TSV file
        val_pairs_path: Optional path to validation pairs TSV file
        model_name: Base transformer model name
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        out_dir: Output directory for checkpoints
        max_length: Maximum sequence length
        use_triplets: Whether to use triplets for training
        triplets_path: Path to triplets TSV file (if use_triplets=True)
    """
    config = get_config()
    device = torch.device(config.DEVICE)

    # Load data
    train_pairs = load_pairs(train_pairs_path)
    if not train_pairs:
        raise ValueError(f"No training pairs found at {train_pairs_path}")

    print(f"Loaded {len(train_pairs)} training pairs")

    # Load triplets if specified
    train_triplets = None
    if use_triplets and triplets_path:
        train_triplets = load_triplets(triplets_path)
        if train_triplets:
            print(f"Loaded {len(train_triplets)} training triplets")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CondenserModel(model_name=model_name)
    model.to(device)

    # Initialize loss functions
    infonce_loss = InfoNCE(temperature=0.05)
    triplet_loss = TripletMarginLoss(margin=0.5) if use_triplets and train_triplets else None

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Create datasets
    if use_triplets and train_triplets:
        train_dataset = TripletsDataset(train_triplets, tokenizer, max_length=max_length)
    else:
        train_dataset = PairsDataset(train_pairs, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
    )

    # Validation dataset (optional)
    val_loader = None
    if val_pairs_path:
        val_pairs = load_pairs(val_pairs_path)
        if val_pairs:
            val_dataset = PairsDataset(val_pairs, tokenizer, max_length=max_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            print(f"Loaded {len(val_pairs)} validation pairs")

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to {run_dir}")

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            # Move batch to device
            if use_triplets and train_triplets:
                # Triplet training
                query_input_ids = batch["query_input_ids"].to(device)
                query_attention_mask = batch["query_attention_mask"].to(device)
                pos_input_ids = batch["pos_input_ids"].to(device)
                pos_attention_mask = batch["pos_attention_mask"].to(device)
                neg_input_ids = batch["neg_input_ids"].to(device)
                neg_attention_mask = batch["neg_attention_mask"].to(device)

                # Encode with shared encoder weights
                query_embeddings = model(query_input_ids, query_attention_mask)
                pos_embeddings = model(pos_input_ids, pos_attention_mask)
                neg_embeddings = model(neg_input_ids, neg_attention_mask)

                # Compute triplet loss
                loss = triplet_loss(query_embeddings, pos_embeddings, neg_embeddings)
            else:
                # Pair training with InfoNCE
                query_input_ids = batch["query_input_ids"].to(device)
                query_attention_mask = batch["query_attention_mask"].to(device)
                doc_input_ids = batch["doc_input_ids"].to(device)
                doc_attention_mask = batch["doc_attention_mask"].to(device)

                # Encode queries and positives with shared encoder weights
                query_embeddings = model(query_input_ids, query_attention_mask)
                doc_embeddings = model(doc_input_ids, doc_attention_mask)

                # Compute similarity matrix via dot product
                # InfoNCE uses in-batch negatives
                loss = infonce_loss(query_embeddings, doc_embeddings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / num_batches})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Validation (optional)
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    query_input_ids = batch["query_input_ids"].to(device)
                    query_attention_mask = batch["query_attention_mask"].to(device)
                    doc_input_ids = batch["doc_input_ids"].to(device)
                    doc_attention_mask = batch["doc_attention_mask"].to(device)

                    query_embeddings = model(query_input_ids, query_attention_mask)
                    doc_embeddings = model(doc_input_ids, doc_attention_mask)

                    loss = infonce_loss(query_embeddings, doc_embeddings)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")
            model.train()

        # Save checkpoint (encoder state_dict and tokenizer)
        checkpoint_path = run_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "model_name": model_name,
            },
            checkpoint_path,
        )

        # Save tokenizer
        tokenizer.save_pretrained(run_dir / "tokenizer")

        print(f"Saved checkpoint to {checkpoint_path}")

    print(f"Training complete! Checkpoints saved to {run_dir}")


if __name__ == "__main__":
    config = get_config()

    parser = argparse.ArgumentParser(description="Train condenser model with contrastive learning")
    parser.add_argument("--train_pairs", type=str, default=str(config.TRAIN_PAIRS_PATH))
    parser.add_argument(
        "--val_pairs",
        type=str,
        default=str(config.VAL_PAIRS_PATH) if config.VAL_PAIRS_PATH.exists() else None,
    )
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--out_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--max_length", type=int, default=config.MAX_LEN)
    parser.add_argument("--triplets", type=str, default=None, help="Path to triplets TSV file (optional)")
    args = parser.parse_args()

    train(
        train_pairs_path=args.train_pairs,
        val_pairs_path=args.val_pairs,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        max_length=args.max_length,
        use_triplets=args.triplets is not None,
        triplets_path=args.triplets,
    )
