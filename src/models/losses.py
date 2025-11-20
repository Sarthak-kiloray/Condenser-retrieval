"""Loss functions for contrastive learning."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for dense retrieval."""

    def __init__(self, temperature: float = 0.05):
        """Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            query_embeddings: Query embeddings [batch_size, dim]
            doc_embeddings: Document embeddings [batch_size, dim]

        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.t()) / self.temperature

        # Positive pairs are on the diagonal
        labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class InBatchNegativesLoss(nn.Module):
    """In-batch negative sampling loss."""

    def __init__(self, temperature: float = 0.05):
        """Initialize in-batch negatives loss.

        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_doc_embeddings: torch.Tensor,
        negative_doc_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute in-batch negatives loss.

        Args:
            query_embeddings: Query embeddings [batch_size, dim]
            positive_doc_embeddings: Positive document embeddings [batch_size, dim]
            negative_doc_embeddings: Optional negative document embeddings

        Returns:
            Loss value
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        positive_doc_embeddings = F.normalize(positive_doc_embeddings, p=2, dim=1)

        # Compute positive similarities
        positive_sim = (query_embeddings * positive_doc_embeddings).sum(dim=1) / self.temperature

        # Use other queries' positives as negatives (in-batch negatives)
        if negative_doc_embeddings is None:
            negative_doc_embeddings = positive_doc_embeddings

        negative_sim = torch.matmul(query_embeddings, negative_doc_embeddings.t()) / self.temperature

        # Combine positive and negative similarities
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class InfoNCE(nn.Module):
    """InfoNCE loss for (q, pos) pairs with in-batch negatives.

    Uses in-batch negatives where other queries' positives serve as negatives.
    """

    def __init__(self, temperature: float = 0.05):
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling similarities
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            query_embeddings: Query embeddings [batch_size, dim]
            positive_embeddings: Positive document embeddings [batch_size, dim]

        Returns:
            InfoNCE loss value
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Compute similarity matrix: [batch_size, batch_size]
        # Each row i has similarity between query i and all positive docs
        # Diagonal elements are positive pairs
        similarity_matrix = torch.matmul(query_embeddings, positive_embeddings.t()) / self.temperature

        # Positive pairs are on the diagonal (query i matches positive doc i)
        labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)

        # Cross-entropy loss: for each query, maximize similarity to its positive
        # while minimizing similarity to other positives (in-batch negatives)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class TripletMarginLoss(nn.Module):
    """Triplet margin loss for (q, pos, neg) triplets."""

    def __init__(self, margin: float = 0.5):
        """Initialize triplet margin loss.

        Args:
            margin: Margin parameter for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet margin loss.

        Args:
            query_embeddings: Query embeddings [batch_size, dim]
            positive_embeddings: Positive document embeddings [batch_size, dim]
            negative_embeddings: Negative document embeddings [batch_size, dim]

        Returns:
            Triplet margin loss value
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

        # Compute distances
        pos_distance = 1 - (query_embeddings * positive_embeddings).sum(dim=1)
        neg_distance = 1 - (query_embeddings * negative_embeddings).sum(dim=1)

        # Triplet loss: max(0, margin + pos_distance - neg_distance)
        loss = F.relu(self.margin + pos_distance - neg_distance).mean()
        return loss

