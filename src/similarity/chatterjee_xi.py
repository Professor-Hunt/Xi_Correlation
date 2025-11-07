"""
Implementation of Chatterjee's rank-based correlation coefficient (xi).

References:
    Chatterjee, S. (2021). A New Coefficient of Correlation.
    Journal of the American Statistical Association, 116(536), 2009-2022.
"""

import numpy as np
from typing import Union, Tuple
import warnings


def chatterjee_xi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Chatterjee's rank correlation coefficient ξ for two vectors.

    The coefficient measures the strength of functional dependence of Y on X.
    It equals 1 when Y is a measurable function of X, and 0 when X and Y
    are independent.

    Parameters
    ----------
    x : np.ndarray
        1D array of shape (n,)
    y : np.ndarray
        1D array of shape (n,)

    Returns
    -------
    float
        The ξ value, typically in range [-ε, 1] where ε is small for finite samples

    Raises
    ------
    ValueError
        If arrays are not 1D or have different lengths

    Notes
    -----
    Time complexity: O(n log n) due to sorting
    Space complexity: O(n)

    For finite samples, ξ can be slightly negative even when variables are independent.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])  # Linear relationship
    >>> chatterjee_xi(x, y)
    1.0

    >>> y_quad = x ** 2  # Nonlinear relationship
    >>> chatterjee_xi(x, y_quad)
    1.0

    >>> y_random = np.random.randn(5)  # Independent
    >>> chatterjee_xi(x, y_random)  # Close to 0
    """
    # Input validation
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"x and y must be 1D arrays, got shapes {x.shape} and {y.shape}")

    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")

    n = len(x)

    if n < 2:
        warnings.warn("Arrays must have at least 2 elements for meaningful correlation")
        return 0.0

    # Sort by x and get the ordering of y
    sorted_idx = np.argsort(x)
    y_sorted = y[sorted_idx]

    # Assign ranks to y_sorted
    # Handle ties by average ranking
    ranks = _rank_with_ties(y_sorted)

    # Compute successive absolute rank differences
    diff = np.abs(np.diff(ranks))

    # Chatterjee's formula
    denominator = n ** 2 - 1
    if denominator == 0:
        return 0.0

    xi = 1 - (3 * np.sum(diff)) / denominator

    return float(xi)


def symmetric_xi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute symmetric version of Chatterjee's xi.

    Since xi(X,Y) measures how well Y can be expressed as a function of X,
    it is asymmetric. The symmetric version is defined as max(ξ(X,Y), ξ(Y,X)).

    Parameters
    ----------
    x : np.ndarray
        1D array
    y : np.ndarray
        1D array

    Returns
    -------
    float
        max(ξ(X,Y), ξ(Y,X))

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> symmetric_xi(x, y)
    1.0
    """
    xi_xy = chatterjee_xi(x, y)
    xi_yx = chatterjee_xi(y, x)
    return max(xi_xy, xi_yx)


def projection_based_xi(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 100,
    random_state: int = None
) -> Tuple[float, float]:
    """
    Compute projection-based xi similarity for sets of high-dimensional embeddings.

    Projects embeddings onto random directions and averages the resulting xi values.
    This approach captures nonlinear dependencies across dimensions.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features)
    Y : np.ndarray
        Array of shape (n_samples, n_features)
    n_projections : int, default=100
        Number of random projections to use
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    mean_xi : float
        Mean xi value across all projections
    std_xi : float
        Standard deviation of xi values across projections

    Raises
    ------
    ValueError
        If X and Y have incompatible shapes

    Notes
    -----
    Time complexity: O(k * n * d + k * n * log(n)) where:
        - k = n_projections
        - n = n_samples
        - d = n_features

    Examples
    --------
    >>> X = np.random.randn(100, 384)  # 100 embeddings of dimension 384
    >>> Y = np.random.randn(100, 384)
    >>> mean_xi, std_xi = projection_based_xi(X, Y, n_projections=50)
    """
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape, got {X.shape} and {Y.shape}")

    if X.ndim != 2:
        raise ValueError(f"X and Y must be 2D arrays, got {X.ndim}D")

    n_samples, n_features = X.shape

    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute correlation")

    # Set random seed
    rng = np.random.RandomState(random_state)

    xi_values = []

    for _ in range(n_projections):
        # Draw random unit vector
        w = rng.randn(n_features)
        w = w / np.linalg.norm(w)

        # Project embeddings
        x_proj = X @ w
        y_proj = Y @ w

        # Compute xi
        xi = chatterjee_xi(x_proj, y_proj)
        xi_values.append(xi)

    xi_values = np.array(xi_values)
    return float(np.mean(xi_values)), float(np.std(xi_values))


def _rank_with_ties(arr: np.ndarray) -> np.ndarray:
    """
    Assign ranks to array elements, handling ties by average ranking.

    Parameters
    ----------
    arr : np.ndarray
        1D array to rank

    Returns
    -------
    np.ndarray
        Array of ranks (1-indexed)
    """
    n = len(arr)
    # Get sorting indices
    sorted_idx = np.argsort(arr)

    # Initialize ranks
    ranks = np.empty(n, dtype=float)

    # Assign ranks
    i = 0
    while i < n:
        j = i
        # Find all elements equal to arr[sorted_idx[i]]
        while j < n and arr[sorted_idx[j]] == arr[sorted_idx[i]]:
            j += 1

        # Assign average rank to all tied elements
        avg_rank = (i + j + 1) / 2  # +1 for 1-indexing
        for k in range(i, j):
            ranks[sorted_idx[k]] = avg_rank

        i = j

    return ranks


def xi_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Convert xi similarity to a distance metric.

    Distance = 1 - xi, so that distance is 0 when xi = 1 (perfect dependence)
    and approaches 1 when xi approaches 0 (independence).

    Parameters
    ----------
    x : np.ndarray
        1D array
    y : np.ndarray
        1D array

    Returns
    -------
    float
        Distance in [0, 1+ε]
    """
    xi = chatterjee_xi(x, y)
    return 1 - xi


def batch_chatterjee_xi(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """
    Compute pairwise xi coefficients for sets of vectors.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_vectors, n_features)
    Y : np.ndarray, optional
        Array of shape (m_vectors, n_features)
        If None, computes pairwise xi for X with itself

    Returns
    -------
    np.ndarray
        Matrix of shape (n_vectors, m_vectors) containing xi coefficients
        If Y is None, returns shape (n_vectors, n_vectors)

    Notes
    -----
    This is useful for computing similarity matrices for clustering or retrieval tasks.

    Examples
    --------
    >>> X = np.random.randn(10, 384)  # 10 embeddings
    >>> xi_matrix = batch_chatterjee_xi(X)  # 10x10 similarity matrix
    >>> xi_matrix.shape
    (10, 10)
    """
    if Y is None:
        Y = X

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")

    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"X and Y must have same number of features, got {X.shape[1]} and {Y.shape[1]}")

    n_x = X.shape[0]
    n_y = Y.shape[0]

    result = np.zeros((n_x, n_y))

    for i in range(n_x):
        for j in range(n_y):
            result[i, j] = chatterjee_xi(X[i], Y[j])

    return result
