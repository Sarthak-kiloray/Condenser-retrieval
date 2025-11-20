"""Pooling strategies for text embeddings."""

import torch
import torch.nn as nn


class CLSPooling(nn.Module):
    """CLS token pooling."""

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool using CLS token.

        Args:
            last_hidden_state: Hidden states from transformer
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        return last_hidden_state[:, 0, :]


class MeanPooling(nn.Module):
    """Mean pooling over sequence length."""

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pool over sequence.

        Args:
            last_hidden_state: Hidden states from transformer
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class MaxPooling(nn.Module):
    """Max pooling over sequence length."""

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Max pool over sequence.

        Args:
            last_hidden_state: Hidden states from transformer
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9
        return torch.max(last_hidden_state, 1)[0]

