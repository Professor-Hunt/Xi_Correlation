"""
Embedding utilities for generating vector representations of text.

Supports multiple models including BERT, Sentence-BERT, and TF-IDF.
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import warnings
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")


class EmbeddingModel:
    """
    Unified interface for various embedding models.

    Supports:
    - Sentence-BERT models (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
    - TF-IDF embeddings
    - LSA (TF-IDF + dimensionality reduction)

    Parameters
    ----------
    model_name : str
        Name of the model to use:
        - 'all-MiniLM-L6-v2' (default, fast and efficient)
        - 'all-mpnet-base-v2' (higher quality, slower)
        - 'tfidf' (TF-IDF vectors)
        - 'lsa' (LSA embeddings)
    cache_dir : str, optional
        Directory to cache downloaded models
    device : str, optional
        Device to use ('cpu', 'cuda', 'mps')

    Examples
    --------
    >>> model = EmbeddingModel('all-MiniLM-L6-v2')
    >>> sentences = ["Hello world", "Good morning"]
    >>> embeddings = model.encode(sentences)
    >>> embeddings.shape
    (2, 384)
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.model = None
        self.vectorizer = None
        self.lsa = None

        self._load_model()

    def _load_model(self):
        """Load the specified model."""
        if self.model_name in ['tfidf', 'lsa']:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn required for TF-IDF/LSA embeddings")
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self._vectorizer_fitted = False
            if self.model_name == 'lsa':
                self.lsa = TruncatedSVD(n_components=100, random_state=42)
                self._lsa_fitted = False
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False
    ) -> np.ndarray:
        """
        Encode sentences into embeddings.

        Parameters
        ----------
        sentences : str or list of str
            Sentence(s) to encode
        batch_size : int, default=32
            Batch size for encoding
        show_progress_bar : bool, default=False
            Whether to show progress bar
        convert_to_numpy : bool, default=True
            Whether to return numpy array (vs torch tensor)
        normalize_embeddings : bool, default=False
            Whether to L2-normalize embeddings

        Returns
        -------
        np.ndarray
            Embeddings of shape (n_sentences, embedding_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        if self.model_name in ['tfidf', 'lsa']:
            # TF-IDF or LSA
            if not self._vectorizer_fitted:
                vectors = self.vectorizer.fit_transform(sentences).toarray()
                self._vectorizer_fitted = True
            else:
                vectors = self.vectorizer.transform(sentences).toarray()

            if self.model_name == 'lsa':
                if not self._lsa_fitted:
                    vectors = self.lsa.fit_transform(vectors)
                    self._lsa_fitted = True
                else:
                    vectors = self.lsa.transform(vectors)
            return vectors
        else:
            # Sentence-BERT
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings
            )
            return embeddings

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Encode queries (alias for encode with query-specific defaults).

        Some models have separate encoders for queries vs documents.
        This method can be overridden for such models.
        """
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self,
        corpus: List[str],
        **kwargs
    ) -> np.ndarray:
        """
        Encode corpus/documents (alias for encode with corpus-specific defaults).
        """
        return self.encode(corpus, **kwargs)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        if self.model_name == 'tfidf':
            return 1000  # max_features
        elif self.model_name == 'lsa':
            return 100  # n_components
        elif self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        else:
            raise RuntimeError("Model not loaded")

    def __repr__(self) -> str:
        return f"EmbeddingModel(model_name='{self.model_name}', dim={self.embedding_dim})"


def load_model(
    model_name: str = 'all-MiniLM-L6-v2',
    **kwargs
) -> EmbeddingModel:
    """
    Convenience function to load an embedding model.

    Parameters
    ----------
    model_name : str
        Model name or identifier
    **kwargs
        Additional arguments passed to EmbeddingModel

    Returns
    -------
    EmbeddingModel
        Loaded model instance

    Examples
    --------
    >>> model = load_model('all-MiniLM-L6-v2')
    >>> embeddings = model.encode(["Hello world"])
    """
    return EmbeddingModel(model_name=model_name, **kwargs)


class EmbeddingCache:
    """
    Cache for storing computed embeddings to avoid recomputation.

    Examples
    --------
    >>> cache = EmbeddingCache()
    >>> model = EmbeddingModel()
    >>> sentences = ["Hello", "World"]
    >>> # First call computes and caches
    >>> emb1 = cache.get_or_compute(sentences, model)
    >>> # Second call retrieves from cache
    >>> emb2 = cache.get_or_compute(sentences, model)
    >>> np.array_equal(emb1, emb2)
    True
    """

    def __init__(self):
        self._cache = {}

    def _get_key(self, sentences: List[str], model_name: str) -> str:
        """
        Generate deterministic cache key.

        Uses SHA256 for consistent hashing across interpreter runs.
        """
        sentences_str = "||".join(sentences)
        content = f"{model_name}::{sentences_str}"
        hash_digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"{model_name}::{hash_digest}"

    def get_or_compute(
        self,
        sentences: List[str],
        model: EmbeddingModel,
        **encode_kwargs
    ) -> np.ndarray:
        """
        Get embeddings from cache or compute if not cached.

        Parameters
        ----------
        sentences : list of str
            Sentences to embed
        model : EmbeddingModel
            Model to use for encoding
        **encode_kwargs
            Additional arguments for encode()

        Returns
        -------
        np.ndarray
            Embeddings
        """
        key = self._get_key(sentences, model.model_name)

        if key not in self._cache:
            self._cache[key] = model.encode(sentences, **encode_kwargs)

        return self._cache[key]

    def clear(self):
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# Pre-defined model configurations
MODELS = {
    'mini': 'all-MiniLM-L6-v2',           # Fast, 384 dim
    'mpnet': 'all-mpnet-base-v2',         # Better quality, 768 dim
    'roberta': 'all-roberta-large-v1',    # Large model, 1024 dim
    'tfidf': 'tfidf',                     # TF-IDF baseline
    'lsa': 'lsa',                         # LSA baseline
}


def get_model_config(alias: str) -> str:
    """
    Get model name from alias.

    Parameters
    ----------
    alias : str
        Model alias ('mini', 'mpnet', etc.)

    Returns
    -------
    str
        Full model name

    Examples
    --------
    >>> get_model_config('mini')
    'all-MiniLM-L6-v2'
    """
    return MODELS.get(alias, alias)
