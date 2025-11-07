"""
Comprehensive similarity metrics for vector embeddings.

Provides implementations and comparisons of multiple similarity measures
including cosine similarity, Chatterjee's xi, Pearson correlation, and Spearman's rho.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from scipy.stats import pearsonr, spearmanr
from .chatterjee_xi import chatterjee_xi, symmetric_xi


def cosine_similarity_score(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    x : np.ndarray
        1D or 2D array (if 2D, must be shape (1, n))
    y : np.ndarray
        1D or 2D array (if 2D, must be shape (1, n))

    Returns
    -------
    float
        Cosine similarity in [-1, 1]

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> cosine_similarity_score(x, y)
    0.9746318461970762
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    return float(sklearn_cosine(x, y)[0, 0])


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        1D array
    y : np.ndarray
        1D array

    Returns
    -------
    correlation : float
        Pearson's r
    p_value : float
        Two-tailed p-value
    """
    r, p = pearsonr(x, y)
    return float(r), float(p)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman's rank correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        1D array
    y : np.ndarray
        1D array

    Returns
    -------
    correlation : float
        Spearman's rho
    p_value : float
        Two-tailed p-value
    """
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def compute_all_similarities(
    x: np.ndarray,
    y: np.ndarray,
    include_pvalues: bool = False
) -> Dict[str, float]:
    """
    Compute all available similarity metrics between two vectors.

    Parameters
    ----------
    x : np.ndarray
        1D array
    y : np.ndarray
        1D array
    include_pvalues : bool, default=False
        Whether to include p-values for statistical tests

    Returns
    -------
    dict
        Dictionary containing all similarity scores:
        - 'cosine': Cosine similarity
        - 'xi': Chatterjee's xi (X->Y)
        - 'xi_symmetric': max(xi(X,Y), xi(Y,X))
        - 'pearson': Pearson's r
        - 'spearman': Spearman's rho
        - 'pearson_pvalue': p-value (if include_pvalues=True)
        - 'spearman_pvalue': p-value (if include_pvalues=True)

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> metrics = compute_all_similarities(x, y)
    >>> print(metrics['cosine'], metrics['xi'])
    """
    results = {
        'cosine': cosine_similarity_score(x, y),
        'xi': chatterjee_xi(x, y),
        'xi_symmetric': symmetric_xi(x, y),
    }

    # Add Pearson correlation
    r, p_pearson = pearson_correlation(x, y)
    results['pearson'] = r
    if include_pvalues:
        results['pearson_pvalue'] = p_pearson

    # Add Spearman correlation
    rho, p_spearman = spearman_correlation(x, y)
    results['spearman'] = rho
    if include_pvalues:
        results['spearman_pvalue'] = p_spearman

    return results


def compute_similarity_matrix(
    embeddings: np.ndarray,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for a set of embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (n_samples, n_features)
    metric : str, default='cosine'
        Similarity metric to use: 'cosine', 'xi', or 'xi_symmetric'

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_samples, n_samples)

    Examples
    --------
    >>> embeddings = np.random.randn(10, 128)
    >>> sim_matrix = compute_similarity_matrix(embeddings, metric='cosine')
    >>> sim_matrix.shape
    (10, 10)
    """
    n = embeddings.shape[0]

    if metric == 'cosine':
        return sklearn_cosine(embeddings)

    # For xi-based metrics, compute pairwise
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif j > i:
                if metric == 'xi':
                    sim_matrix[i, j] = chatterjee_xi(embeddings[i], embeddings[j])
                elif metric == 'xi_symmetric':
                    sim_matrix[i, j] = symmetric_xi(embeddings[i], embeddings[j])
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                sim_matrix[j, i] = sim_matrix[i, j]  # Make symmetric

    return sim_matrix


def rank_by_similarity(
    query: np.ndarray,
    documents: np.ndarray,
    metric: str = 'cosine',
    top_k: Optional[int] = None
) -> List[Tuple[int, float]]:
    """
    Rank documents by similarity to a query.

    Parameters
    ----------
    query : np.ndarray
        Query vector of shape (n_features,)
    documents : np.ndarray
        Document vectors of shape (n_docs, n_features)
    metric : str, default='cosine'
        Similarity metric: 'cosine', 'xi', or 'xi_symmetric'
    top_k : int, optional
        Return only top k results. If None, return all.

    Returns
    -------
    list of (int, float)
        List of (document_index, similarity_score) tuples, sorted by score descending

    Examples
    --------
    >>> query = np.random.randn(128)
    >>> documents = np.random.randn(100, 128)
    >>> rankings = rank_by_similarity(query, documents, metric='cosine', top_k=5)
    >>> top_doc_idx, top_score = rankings[0]
    """
    n_docs = documents.shape[0]
    similarities = []

    for i in range(n_docs):
        if metric == 'cosine':
            score = cosine_similarity_score(query, documents[i])
        elif metric == 'xi':
            score = chatterjee_xi(query, documents[i])
        elif metric == 'xi_symmetric':
            score = symmetric_xi(query, documents[i])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        similarities.append((i, score))

    # Sort by score descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        similarities = similarities[:top_k]

    return similarities


class SimilarityMetrics:
    """
    Convenience class for computing multiple similarity metrics.

    Examples
    --------
    >>> sm = SimilarityMetrics()
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> results = sm.compute_all(x, y)
    >>> print(results)
    """

    def __init__(self):
        self.metrics = ['cosine', 'xi', 'xi_symmetric', 'pearson', 'spearman']

    def compute_all(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute all available metrics."""
        return compute_all_similarities(x, y, include_pvalues=False)

    def compute_batch(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for multiple pairs of vectors.

        Parameters
        ----------
        pairs : list of (np.ndarray, np.ndarray)
            List of vector pairs

        Returns
        -------
        list of dict
            List of metric dictionaries
        """
        return [self.compute_all(x, y) for x, y in pairs]

    def compare_metrics(
        self,
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute and optionally print all metrics for comparison.

        Parameters
        ----------
        x : np.ndarray
            First vector
        y : np.ndarray
            Second vector
        verbose : bool, default=True
            Whether to print results

        Returns
        -------
        dict
            All computed metrics
        """
        results = self.compute_all(x, y)

        if verbose:
            print("Similarity Metrics Comparison")
            print("-" * 40)
            for metric, value in results.items():
                print(f"{metric:20s}: {value:8.4f}")

        return results
