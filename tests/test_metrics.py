"""Tests for similarity metrics module."""

import pytest
import numpy as np
from src.similarity.metrics import (
    cosine_similarity_score,
    compute_all_similarities,
    compute_similarity_matrix,
    rank_by_similarity,
    SimilarityMetrics
)


class TestCosineSimilarity:
    """Test cases for cosine similarity."""

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        x = np.array([1, 2, 3], dtype=float)
        y = np.array([1, 2, 3], dtype=float)
        cos_sim = cosine_similarity_score(x, y)
        assert abs(cos_sim - 1.0) < 1e-10, f"Expected 1.0, got {cos_sim}"

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        x = np.array([1, 0, 0], dtype=float)
        y = np.array([0, 1, 0], dtype=float)
        cos_sim = cosine_similarity_score(x, y)
        assert abs(cos_sim) < 1e-10, f"Expected 0.0, got {cos_sim}"

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        x = np.array([1, 2, 3], dtype=float)
        y = np.array([-1, -2, -3], dtype=float)
        cos_sim = cosine_similarity_score(x, y)
        assert abs(cos_sim - (-1.0)) < 1e-10, f"Expected -1.0, got {cos_sim}"

    def test_2d_input(self):
        """Test cosine similarity with 2D input."""
        x = np.array([[1, 2, 3]], dtype=float)
        y = np.array([[1, 2, 3]], dtype=float)
        cos_sim = cosine_similarity_score(x, y)
        assert abs(cos_sim - 1.0) < 1e-10


class TestComputeAllSimilarities:
    """Test cases for compute_all_similarities."""

    def test_returns_all_metrics(self):
        """Test that all metrics are computed."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + 0.1 * np.random.randn(100)

        metrics = compute_all_similarities(x, y)

        expected_keys = ['cosine', 'xi', 'xi_symmetric', 'pearson', 'spearman']
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float)

    def test_with_pvalues(self):
        """Test computation with p-values."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + 0.1 * np.random.randn(100)

        metrics = compute_all_similarities(x, y, include_pvalues=True)

        assert 'pearson_pvalue' in metrics
        assert 'spearman_pvalue' in metrics
        assert 0 <= metrics['pearson_pvalue'] <= 1
        assert 0 <= metrics['spearman_pvalue'] <= 1

    def test_on_perfect_correlation(self):
        """Test metrics on perfectly correlated data.

        Note: For small samples (n=5), xi ≈ 0.5 even for perfect correlation.
        This is expected finite-sample behavior per Chatterjee (2021).
        """
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * x + 1

        metrics = compute_all_similarities(x, y)

        assert metrics['cosine'] > 0.99, "Cosine should be near 1"
        assert metrics['pearson'] > 0.99, "Pearson should be near 1"
        # For n=5, xi ≈ 0.5 is expected for perfect functional relationships
        assert 0.4 <= metrics['xi'] <= 0.6, f"Xi should be ≈ 0.5 for n=5, got {metrics['xi']}"


class TestComputeSimilarityMatrix:
    """Test cases for compute_similarity_matrix."""

    def test_cosine_matrix(self):
        """Test cosine similarity matrix."""
        np.random.seed(42)
        embeddings = np.random.randn(5, 10)

        matrix = compute_similarity_matrix(embeddings, metric='cosine')

        assert matrix.shape == (5, 5)
        # Diagonal should be 1
        for i in range(5):
            assert abs(matrix[i, i] - 1.0) < 1e-6
        # Should be symmetric
        assert np.allclose(matrix, matrix.T)

    def test_xi_matrix(self):
        """Test xi similarity matrix."""
        np.random.seed(42)
        embeddings = np.random.randn(5, 20)

        matrix = compute_similarity_matrix(embeddings, metric='xi')

        assert matrix.shape == (5, 5)
        assert np.allclose(matrix, matrix.T), "Matrix should be symmetric"

    def test_invalid_metric(self):
        """Test with invalid metric name."""
        embeddings = np.random.randn(3, 10)

        with pytest.raises(ValueError):
            compute_similarity_matrix(embeddings, metric='invalid_metric')


class TestRankBySimilarity:
    """Test cases for rank_by_similarity."""

    def test_basic_ranking(self):
        """Test basic document ranking."""
        np.random.seed(42)
        query = np.random.randn(10)
        documents = np.random.randn(5, 10)

        rankings = rank_by_similarity(query, documents, metric='cosine')

        assert len(rankings) == 5
        # Check format
        for idx, score in rankings:
            assert isinstance(idx, (int, np.integer))
            assert isinstance(score, (float, np.floating))

        # Check descending order
        scores = [score for _, score in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_ranking(self):
        """Test ranking with top_k parameter."""
        np.random.seed(42)
        query = np.random.randn(10)
        documents = np.random.randn(10, 10)

        rankings = rank_by_similarity(query, documents, metric='cosine', top_k=3)

        assert len(rankings) == 3

    def test_ranking_with_xi(self):
        """Test ranking using xi metric."""
        np.random.seed(42)
        query = np.random.randn(20)
        documents = np.random.randn(5, 20)

        rankings = rank_by_similarity(query, documents, metric='xi', top_k=5)

        assert len(rankings) == 5
        # Verify descending order
        scores = [score for _, score in rankings]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


class TestSimilarityMetrics:
    """Test cases for SimilarityMetrics class."""

    def test_initialization(self):
        """Test SimilarityMetrics initialization."""
        sm = SimilarityMetrics()
        assert hasattr(sm, 'metrics')
        assert len(sm.metrics) > 0

    def test_compute_all(self):
        """Test compute_all method."""
        sm = SimilarityMetrics()
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)

        results = sm.compute_all(x, y)

        assert isinstance(results, dict)
        assert 'cosine' in results
        assert 'xi' in results

    def test_compute_batch(self):
        """Test compute_batch method."""
        sm = SimilarityMetrics()

        pairs = [
            (np.array([1, 2, 3], dtype=float), np.array([4, 5, 6], dtype=float)),
            (np.array([7, 8, 9], dtype=float), np.array([10, 11, 12], dtype=float))
        ]

        results = sm.compute_batch(pairs)

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_compare_metrics(self):
        """Test compare_metrics method."""
        sm = SimilarityMetrics()
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = x ** 2

        results = sm.compare_metrics(x, y, verbose=False)

        assert isinstance(results, dict)
        assert 'cosine' in results
        assert 'xi' in results


class TestEdgeCases:
    """Test edge cases for metrics."""

    def test_zero_vectors(self):
        """Test with zero vectors."""
        x = np.zeros(10)
        y = np.ones(10)

        # Cosine should handle this (may return 0 or NaN)
        try:
            cos_sim = cosine_similarity_score(x, y)
            assert np.isnan(cos_sim) or cos_sim == 0
        except (ValueError, RuntimeWarning):
            pass

    def test_single_element_vectors(self):
        """Test with single element vectors."""
        x = np.array([1.0])
        y = np.array([2.0])

        # Should work or raise appropriate error
        try:
            metrics = compute_all_similarities(x, y)
            assert isinstance(metrics, dict)
        except ValueError:
            # Acceptable to reject single-element vectors
            pass

    def test_very_small_values(self):
        """Test with very small values."""
        x = np.array([1e-10, 2e-10, 3e-10])
        y = np.array([4e-10, 5e-10, 6e-10])

        metrics = compute_all_similarities(x, y)

        # Should not produce NaN or Inf
        for key, value in metrics.items():
            assert not np.isnan(value), f"{key} produced NaN"
            assert not np.isinf(value), f"{key} produced Inf"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
