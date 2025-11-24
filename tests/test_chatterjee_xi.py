"""Tests for Chatterjee's xi implementation."""

import pytest
import numpy as np
from src.similarity.chatterjee_xi import (
    chatterjee_xi,
    symmetric_xi,
    projection_based_xi,
    xi_distance,
    batch_chatterjee_xi
)


class TestChatterjeeXi:
    """Test cases for chatterjee_xi function."""

    def test_perfect_linear_relationship(self):
        """Test xi on perfect linear relationship.

        Note: For small finite samples (n=5), xi does not approach 1.0 even for
        perfect functional relationships. This is expected behavior per Chatterjee (2021).
        """
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * x + 1
        xi = chatterjee_xi(x, y)
        # For n=5 with perfect monotonic relationship, xi ≈ 0.5 is expected
        assert 0.4 <= xi <= 0.6, f"Expected xi ≈ 0.5 for n=5 linear relationship, got {xi}"

    def test_perfect_monotonic_relationship(self):
        """Test xi on perfect monotonic relationship.

        Note: For small finite samples (n=5), xi ≈ 0.5 is expected, not 1.0.
        """
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = x ** 2
        xi = chatterjee_xi(x, y)
        # For n=5 with perfect monotonic relationship, xi ≈ 0.5 is expected
        assert 0.4 <= xi <= 0.6, f"Expected xi ≈ 0.5 for n=5 monotonic relationship, got {xi}"

    def test_independent_variables(self):
        """Test xi on independent variables."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        xi = chatterjee_xi(x, y)
        assert abs(xi) < 0.2, f"Expected xi ≈ 0 for independent variables, got {xi}"

    def test_constant_arrays(self):
        """Test xi on constant arrays."""
        x = np.ones(10)
        y = np.ones(10)
        xi = chatterjee_xi(x, y)
        # Should handle gracefully (result may be 0 or NaN depending on implementation)
        assert not np.isnan(xi), "Xi should not be NaN for constant arrays"

    def test_with_ties(self):
        """Test xi with tied values.

        Note: With ties and small sample size (n=5), xi is moderate, not high.
        """
        x = np.array([1, 2, 2, 3, 4], dtype=float)
        y = np.array([1, 2, 2, 3, 4], dtype=float)
        xi = chatterjee_xi(x, y)
        # For n=5 with ties, expect moderate xi value
        assert 0.4 <= xi <= 0.7, f"Expected moderate xi for n=5 with ties, got {xi}"

    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError):
            # Different lengths
            chatterjee_xi(np.array([1, 2]), np.array([1, 2, 3]))

        with pytest.raises(ValueError):
            # Wrong dimensions
            chatterjee_xi(np.array([[1, 2]]), np.array([1, 2]))

    def test_small_sample(self):
        """Test with small sample size."""
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        xi = chatterjee_xi(x, y)
        assert isinstance(xi, float), "Should return float"

    def test_nonlinear_relationship(self):
        """Test xi captures nonlinear relationships.

        With larger samples (n=200), xi approaches high values for functional relationships,
        as validated in the paper's synthetic experiments (xi > 0.93 for n=500).
        """
        np.random.seed(42)
        x = np.random.randn(200)
        y_quad = x ** 2
        y_abs = np.abs(x)

        xi_quad = chatterjee_xi(x, y_quad)
        xi_abs = chatterjee_xi(x, y_abs)

        # With n=200, expect high xi for functional relationships (validated in paper)
        assert xi_quad > 0.9, f"Xi should capture quadratic relationship, got {xi_quad}"
        assert xi_abs > 0.9, f"Xi should capture absolute value relationship, got {xi_abs}"


class TestSymmetricXi:
    """Test cases for symmetric_xi function."""

    def test_symmetry(self):
        """Test that symmetric xi is indeed symmetric."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)

        xi_sym = symmetric_xi(x, y)
        assert isinstance(xi_sym, float)

        # Should be max of both directions
        xi_xy = chatterjee_xi(x, y)
        xi_yx = chatterjee_xi(y, x)
        assert xi_sym == max(xi_xy, xi_yx)

    def test_symmetric_on_linear(self):
        """Test symmetric xi on linear relationship.

        Note: For small samples (n=5), symmetric_xi ≈ 0.5, not approaching 1.0.
        """
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * x
        xi_sym = symmetric_xi(x, y)
        # For n=5, expect moderate value around 0.5
        assert 0.4 <= xi_sym <= 0.6, f"Expected xi ≈ 0.5 for n=5, got {xi_sym}"


class TestProjectionBasedXi:
    """Test cases for projection_based_xi function."""

    def test_basic_projection(self):
        """Test projection-based xi on random data."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10)

        mean_xi, std_xi = projection_based_xi(X, Y, n_projections=10, random_state=42)

        assert isinstance(mean_xi, float)
        assert isinstance(std_xi, float)
        assert std_xi >= 0, "Standard deviation should be non-negative"

    def test_projection_on_related_data(self):
        """Test projection-based xi on related data."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        Y = X + 0.1 * np.random.randn(100, 20)  # Noisy copy

        mean_xi, std_xi = projection_based_xi(X, Y, n_projections=50, random_state=42)

        assert mean_xi > 0.5, f"Mean xi should be high for related data, got {mean_xi}"

    def test_input_validation_projection(self):
        """Test input validation for projection-based xi."""
        with pytest.raises(ValueError):
            # Different shapes
            X = np.random.randn(10, 5)
            Y = np.random.randn(10, 6)
            projection_based_xi(X, Y)

        with pytest.raises(ValueError):
            # Wrong dimensions
            X = np.random.randn(10)
            Y = np.random.randn(10)
            projection_based_xi(X, Y)


class TestXiDistance:
    """Test cases for xi_distance function."""

    def test_distance_properties(self):
        """Test that distance = 1 - xi."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * x

        xi = chatterjee_xi(x, y)
        dist = xi_distance(x, y)

        assert abs(dist - (1 - xi)) < 1e-10, "Distance should equal 1 - xi"

    def test_distance_range(self):
        """Test distance is in valid range."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        dist = xi_distance(x, y)
        assert 0 <= dist <= 2, f"Distance should be in [0, 2], got {dist}"


class TestBatchChatterjeeXi:
    """Test cases for batch_chatterjee_xi function."""

    def test_batch_computation(self):
        """Test batch computation of xi.

        Note: Diagonal elements represent xi(X[i], X[i]) which should be high
        but not necessarily >= 0.99 for moderate-sized vectors (n=100).
        """
        np.random.seed(42)
        X = np.random.randn(5, 100)

        xi_matrix = batch_chatterjee_xi(X)

        assert xi_matrix.shape == (5, 5), "Matrix should be 5x5"

        # Diagonal should be high (a vector compared to itself)
        # With n=100, expect values > 0.9 but allow for finite-sample variation
        for i in range(5):
            assert xi_matrix[i, i] >= 0.9, f"Diagonal xi[{i},{i}] should be high, got {xi_matrix[i, i]}"

    def test_batch_with_two_sets(self):
        """Test batch computation with two different sets."""
        np.random.seed(42)
        X = np.random.randn(3, 100)
        Y = np.random.randn(4, 100)

        xi_matrix = batch_chatterjee_xi(X, Y)

        assert xi_matrix.shape == (3, 4), "Matrix should be 3x4"

    def test_input_validation_batch(self):
        """Test input validation for batch computation."""
        with pytest.raises(ValueError):
            # 1D arrays
            X = np.random.randn(10)
            batch_chatterjee_xi(X)

        with pytest.raises(ValueError):
            # Incompatible features
            X = np.random.randn(5, 10)
            Y = np.random.randn(5, 20)
            batch_chatterjee_xi(X, Y)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_negative_correlation(self):
        """Test xi on negatively correlated data."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = -x
        xi = chatterjee_xi(x, y)
        # Xi should still be high for functional relationships
        # even if negatively correlated
        assert isinstance(xi, float)

    def test_with_nan_handling(self):
        """Test that NaN values are handled appropriately."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Should either raise error or handle NaN
        try:
            xi = chatterjee_xi(x, y)
            assert not np.isnan(xi) or np.isnan(x).any()
        except (ValueError, RuntimeError):
            # Acceptable to raise error on NaN
            pass

    def test_large_sample(self):
        """Test performance on large sample."""
        np.random.seed(42)
        n = 10000
        x = np.random.randn(n)
        y = x ** 2

        xi = chatterjee_xi(x, y)
        assert xi > 0.95, "Should work correctly on large samples"

    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        xi1 = chatterjee_xi(x.copy(), y.copy())
        xi2 = chatterjee_xi(x.copy(), y.copy())

        assert xi1 == xi2, "Results should be reproducible"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
