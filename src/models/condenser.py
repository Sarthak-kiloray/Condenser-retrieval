"""Condenser model implementation."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CondenserModel(nn.Module):
    """Condenser-style dense retrieval model."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        pooling: str = "cls",
    ):
        """Initialize Condenser model.

        Args:
            model_name: Base transformer model name
            embedding_dim: Embedding dimension
            pooling: Pooling strategy ('cls', 'mean', 'max')
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.pooling = pooling

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode(self, input_ids, attention_mask):
        """Encode input tokens.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Encoded embeddings
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        if self.pooling == "cls":
            embeddings = last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            embeddings = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
                1, keepdim=True
            )
        else:
            embeddings = last_hidden_state.max(dim=1)[0]

        return self.projection(embeddings)

    def forward(self, input_ids, attention_mask):
        """Forward pass.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Encoded embeddings
        """
        return self.encode(input_ids, attention_mask)

