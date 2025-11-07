"""
Vector Correlation: A Research Project on Semantic Similarity
Using Chatterjee's Xi Correlation Coefficient

This package provides tools for measuring semantic similarity using
rank-based correlation methods as an alternative to cosine similarity.
"""

__version__ = "1.0.0"
__author__ = "Professor Hunt"

from .similarity.chatterjee_xi import chatterjee_xi, symmetric_xi
from .similarity.metrics import compute_all_similarities, cosine_similarity_score

__all__ = [
    "chatterjee_xi",
    "symmetric_xi",
    "compute_all_similarities",
    "cosine_similarity_score",
]
