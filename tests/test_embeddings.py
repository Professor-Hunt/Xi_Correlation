"""Tests for embedding models."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.similarity.embeddings import (
    EmbeddingModel,
    load_model,
    EmbeddingCache,
    get_model_config,
    MODELS
)


class TestEmbeddingModel:
    """Test cases for EmbeddingModel class."""

    @pytest.mark.skipif(True, reason="Requires sentence-transformers, skip for CI")
    def test_load_mini_model(self):
        """Test loading MiniLM model."""
        model = EmbeddingModel('all-MiniLM-L6-v2')
        assert model.model is not None
        assert model.embedding_dim == 384

    def test_model_repr(self):
        """Test string representation."""
        model = EmbeddingModel('tfidf')
        repr_str = repr(model)
        assert 'tfidf' in repr_str
        assert 'dim' in repr_str.lower()

    @pytest.mark.skipif(True, reason="Requires sentence-transformers, skip for CI")
    def test_encode_single_sentence(self):
        """Test encoding a single sentence."""
        model = EmbeddingModel('all-MiniLM-L6-v2')
        embedding = model.encode("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 384)

    @pytest.mark.skipif(True, reason="Requires sentence-transformers, skip for CI")
    def test_encode_multiple_sentences(self):
        """Test encoding multiple sentences."""
        model = EmbeddingModel('all-MiniLM-L6-v2')
        sentences = ["Hello world", "Good morning", "How are you"]
        embeddings = model.encode(sentences)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_tfidf_embeddings(self):
        """Test TF-IDF embeddings."""
        model = EmbeddingModel('tfidf')
        sentences = ["Hello world", "Good morning world", "Hello there"]
        embeddings = model.encode(sentences)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] <= 1000  # max_features

    def test_lsa_embeddings(self):
        """Test LSA embeddings."""
        model = EmbeddingModel('lsa')
        sentences = ["Hello world", "Good morning world", "Hello there"] * 10
        embeddings = model.encode(sentences)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 30
        assert embeddings.shape[1] == 100  # n_components

    def test_embedding_dim_property(self):
        """Test embedding_dim property."""
        model_tfidf = EmbeddingModel('tfidf')
        assert model_tfidf.embedding_dim == 1000

        model_lsa = EmbeddingModel('lsa')
        assert model_lsa.embedding_dim == 100


class TestLoadModel:
    """Test cases for load_model function."""

    def test_load_model_tfidf(self):
        """Test loading model via load_model function."""
        model = load_model('tfidf')
        assert isinstance(model, EmbeddingModel)
        assert model.model_name == 'tfidf'

    def test_load_model_with_kwargs(self):
        """Test loading model with additional arguments."""
        model = load_model('tfidf', device='cpu')
        assert isinstance(model, EmbeddingModel)


class TestEmbeddingCache:
    """Test cases for EmbeddingCache."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = EmbeddingCache()
        assert len(cache) == 0

    def test_cache_miss_and_hit(self):
        """Test cache miss and subsequent hit."""
        cache = EmbeddingCache()
        model = EmbeddingModel('tfidf')
        sentences = ["Hello world", "Good morning"]

        # First call - cache miss
        emb1 = cache.get_or_compute(sentences, model)
        assert len(cache) == 1

        # Second call - cache hit
        emb2 = cache.get_or_compute(sentences, model)
        assert len(cache) == 1  # No new entry

        # Should return same embeddings
        assert np.array_equal(emb1, emb2)

    def test_cache_different_sentences(self):
        """Test cache with different sentences."""
        cache = EmbeddingCache()
        model = EmbeddingModel('tfidf')

        sentences1 = ["Hello world"]
        sentences2 = ["Good morning"]

        emb1 = cache.get_or_compute(sentences1, model)
        emb2 = cache.get_or_compute(sentences2, model)

        assert len(cache) == 2  # Two different entries

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = EmbeddingCache()
        model = EmbeddingModel('tfidf')

        cache.get_or_compute(["Hello"], model)
        assert len(cache) == 1

        cache.clear()
        assert len(cache) == 0


class TestGetModelConfig:
    """Test cases for get_model_config function."""

    def test_get_config_by_alias(self):
        """Test getting config by alias."""
        config = get_model_config('mini')
        assert config == 'all-MiniLM-L6-v2'

        config = get_model_config('tfidf')
        assert config == 'tfidf'

    def test_get_config_full_name(self):
        """Test passing full model name."""
        config = get_model_config('all-mpnet-base-v2')
        assert config == 'all-mpnet-base-v2'

    def test_models_dict(self):
        """Test MODELS dictionary."""
        assert 'mini' in MODELS
        assert 'tfidf' in MODELS
        assert 'lsa' in MODELS


class TestEdgeCases:
    """Test edge cases for embeddings."""

    def test_empty_sentence(self):
        """Test encoding empty sentence."""
        model = EmbeddingModel('tfidf')

        try:
            embeddings = model.encode([""])
            assert embeddings.shape[0] == 1
        except ValueError:
            # Acceptable to reject empty sentences
            pass

    def test_very_long_sentence(self):
        """Test encoding very long sentence."""
        model = EmbeddingModel('tfidf')
        long_sentence = "word " * 1000

        embeddings = model.encode([long_sentence])
        assert embeddings.shape[0] == 1

    def test_special_characters(self):
        """Test encoding sentences with special characters."""
        model = EmbeddingModel('tfidf')
        sentences = ["Hello!!! @#$%", "Test... ???", "UTF-8: 你好"]

        embeddings = model.encode(sentences)
        assert embeddings.shape[0] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
