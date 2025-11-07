"""
Similarity measures module.

Provides implementations of various similarity metrics including
Chatterjee's xi, cosine similarity, and projection-based variants.
"""

from .chatterjee_xi import chatterjee_xi, symmetric_xi, projection_based_xi
from .metrics import compute_all_similarities, cosine_similarity_score
from .embeddings import EmbeddingModel

__all__ = [
    "chatterjee_xi",
    "symmetric_xi",
    "projection_based_xi",
    "compute_all_similarities",
    "cosine_similarity_score",
    "EmbeddingModel",
]
