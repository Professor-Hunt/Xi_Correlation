"""
Stochastic embedding generation for projection-based Xi computation.

This module addresses the methodological requirement that projection-based Xi
needs multiple observations (n_samples, n_features) rather than single vectors.

For comparing two sentences A and B, we generate n stochastic embeddings for each:
- X: (n, d) embeddings for sentence A
- Y: (n, d) embeddings for sentence B

Then compute projection_based_xi(X, Y) to get a statistically valid similarity score.
"""

import numpy as np
import warnings
from typing import List, Tuple, Optional
from pathlib import Path

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Stochastic embeddings will use fallback methods.")


class StochasticEmbedder:
    """
    Generate multiple stochastic embeddings for the same input.

    This is required for the projection-based Xi method, which needs
    (n_samples, n_features) shaped arrays rather than single vectors.

    Methods:
    - Dropout-based: Run inference with dropout enabled
    - Perturbation-based: Add small random noise to deterministic embeddings

    Parameters
    ----------
    model_name : str
        Name of the sentence-transformers model
    method : str, default='dropout'
        Method for generating stochastic embeddings: 'dropout' or 'perturbation'
    dropout_rate : float, default=0.1
        Dropout rate for 'dropout' method
    perturbation_std : float, default=0.01
        Standard deviation for 'perturbation' method

    Examples
    --------
    >>> embedder = StochasticEmbedder('all-MiniLM-L6-v2', method='dropout')
    >>> X, Y = embedder.encode_pair(
    ...     "The cat sat on the mat",
    ...     "A feline rested on the rug",
    ...     n_samples=50
    ... )
    >>> X.shape  # (50, 384) - 50 stochastic embeddings
    >>> Y.shape  # (50, 384)
    >>>
    >>> # Now compute projection-based Xi
    >>> from src.similarity.chatterjee_xi import projection_based_xi
    >>> mean_xi, std_xi = projection_based_xi(X, Y, n_projections=100)
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        method: str = 'dropout',
        dropout_rate: float = 0.1,
        perturbation_std: float = 0.01,
        device: Optional[str] = None,
        local_model_path: Optional[str] = None
    ):
        self.model_name = model_name
        self.method = method
        self.dropout_rate = dropout_rate
        self.perturbation_std = perturbation_std

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for StochasticEmbedder. "
                "Install with: pip install sentence-transformers"
            )

        # Load model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Determine model path: use local_model_path if provided,
        # or if model_name is a local path, or download from HuggingFace
        if local_model_path:
            model_path = local_model_path
        elif Path(model_name).exists():
            model_path = model_name
        else:
            model_path = model_name

        self.model = SentenceTransformer(model_path, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode_stochastic_dropout(
        self,
        sentence: str,
        n_samples: int = 50
    ) -> np.ndarray:
        """
        Generate stochastic embeddings using dropout.

        Runs inference multiple times with dropout enabled to get
        different embeddings for the same input.

        Parameters
        ----------
        sentence : str
            Input sentence
        n_samples : int
            Number of stochastic samples to generate

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, embedding_dim)
        """
        # Enable dropout
        self.model.train()

        embeddings = []
        with torch.no_grad():
            for _ in range(n_samples):
                emb = self.model.encode(sentence, convert_to_numpy=True)
                embeddings.append(emb)

        # Restore eval mode
        self.model.eval()

        return np.array(embeddings)

    def encode_stochastic_perturbation(
        self,
        sentence: str,
        n_samples: int = 50
    ) -> np.ndarray:
        """
        Generate stochastic embeddings using perturbation.

        Computes one deterministic embedding and adds small random noise
        to create multiple samples. This is faster than dropout but less
        principled.

        Parameters
        ----------
        sentence : str
            Input sentence
        n_samples : int
            Number of stochastic samples to generate

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, embedding_dim)
        """
        # Get deterministic embedding
        self.model.eval()
        base_emb = self.model.encode(sentence, convert_to_numpy=True)

        # Generate perturbations
        perturbations = np.random.normal(
            loc=0,
            scale=self.perturbation_std,
            size=(n_samples, self.embedding_dim)
        )

        embeddings = base_emb + perturbations

        # Normalize to unit vectors (maintain on unit sphere)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings

    def encode_single(
        self,
        sentence: str,
        n_samples: int = 50
    ) -> np.ndarray:
        """
        Generate n_samples stochastic embeddings for a single sentence.

        Parameters
        ----------
        sentence : str
            Input sentence
        n_samples : int
            Number of stochastic samples to generate

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, embedding_dim)
        """
        if self.method == 'dropout':
            return self.encode_stochastic_dropout(sentence, n_samples)
        elif self.method == 'perturbation':
            return self.encode_stochastic_perturbation(sentence, n_samples)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def encode_pair(
        self,
        sentence1: str,
        sentence2: str,
        n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stochastic embeddings for a sentence pair.

        This is the primary method for preparing data for projection-based Xi.

        Parameters
        ----------
        sentence1, sentence2 : str
            Input sentences to compare
        n_samples : int
            Number of stochastic samples per sentence

        Returns
        -------
        X, Y : np.ndarray
            Arrays of shape (n_samples, embedding_dim)

        Examples
        --------
        >>> embedder = StochasticEmbedder('all-MiniLM-L6-v2')
        >>> X, Y = embedder.encode_pair(
        ...     "The cat sat on the mat",
        ...     "A feline rested on the rug",
        ...     n_samples=50
        ... )
        >>> from src.similarity.chatterjee_xi import projection_based_xi
        >>> mean_xi, std_xi = projection_based_xi(X, Y)
        """
        X = self.encode_single(sentence1, n_samples)
        Y = self.encode_single(sentence2, n_samples)
        return X, Y

    def encode_batch_pairs(
        self,
        sentence_pairs: List[Tuple[str, str]],
        n_samples: int = 50,
        verbose: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate stochastic embeddings for multiple sentence pairs.

        Parameters
        ----------
        sentence_pairs : list of (str, str)
            List of sentence pairs
        n_samples : int
            Number of stochastic samples per sentence
        verbose : bool
            Whether to print progress

        Returns
        -------
        list of (np.ndarray, np.ndarray)
            List of (X, Y) embedding arrays
        """
        results = []
        for i, (sent1, sent2) in enumerate(sentence_pairs):
            if verbose and i % 10 == 0:
                print(f"Processing pair {i+1}/{len(sentence_pairs)}...")

            X, Y = self.encode_pair(sent1, sent2, n_samples)
            results.append((X, Y))

        return results


def verify_projection_based_method():
    """
    Verification script to demonstrate projection-based method.

    This shows the correct usage of projection_based_xi with
    stochastically generated embeddings.
    """
    print("=" * 70)
    print("VERIFICATION: Projection-Based Xi Method")
    print("=" * 70)

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\nSentence-transformers not available. Using synthetic data.\n")

        # Synthetic demonstration
        n_samples = 100
        dim = 384

        # Generate synthetic embeddings with a nonlinear relationship
        X = np.random.randn(n_samples, dim)
        # Y has a nonlinear relationship with X
        Y = np.sign(X) * (X ** 2) + np.random.randn(n_samples, dim) * 0.1

        print(f"Generated synthetic data:")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")

    else:
        print("\nGenerating stochastic BERT embeddings...\n")

        embedder = StochasticEmbedder('all-MiniLM-L6-v2', method='dropout')

        X, Y = embedder.encode_pair(
            "The cat sat on the mat",
            "A feline rested on the rug",
            n_samples=50
        )

        print(f"Generated stochastic embeddings:")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")

    # Compute projection-based Xi
    from ..similarity.chatterjee_xi import projection_based_xi

    mean_xi, std_xi = projection_based_xi(X, Y, n_projections=100)

    print(f"\nProjection-based Xi results:")
    print(f"  Mean Xi: {mean_xi:.4f}")
    print(f"  Std Xi: {std_xi:.4f}")

    print("\n" + "=" * 70)
    print("✓ Verification complete")
    print("=" * 70)


if __name__ == "__main__":
    verify_projection_based_method()
