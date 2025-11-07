"""
Hybrid similarity combining cosine and Chatterjee's Xi.

This module implements weighted combinations of cosine similarity and
symmetrized Xi to leverage the strengths of both metrics.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr

from .metrics import cosine_similarity_score as cosine_similarity
from .chatterjee_xi import symmetric_xi as chatterjee_xi_symmetric


def hybrid_similarity(
    x: np.ndarray,
    y: np.ndarray,
    weight_cosine: float = 0.5,
    weight_xi: Optional[float] = None
) -> float:
    """
    Compute weighted hybrid similarity combining cosine and symmetric Xi.

    Parameters
    ----------
    x, y : np.ndarray
        Input vectors
    weight_cosine : float, default=0.5
        Weight for cosine similarity (0 to 1)
    weight_xi : float, optional
        Weight for Xi. If None, set to (1 - weight_cosine)

    Returns
    -------
    float
        Hybrid similarity score

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> hybrid_similarity(x, y, weight_cosine=0.7)  # 70% cosine, 30% xi
    """
    if weight_xi is None:
        weight_xi = 1.0 - weight_cosine

    # Normalize weights
    total = weight_cosine + weight_xi
    w_cos = weight_cosine / total
    w_xi = weight_xi / total

    cos_sim = cosine_similarity(x, y)
    xi_sim = chatterjee_xi_symmetric(x, y)

    return w_cos * cos_sim + w_xi * xi_sim


def optimize_hybrid_weights(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    labels: np.ndarray,
    metric: str = 'accuracy'
) -> Tuple[float, dict]:
    """
    Find optimal weight for cosine in hybrid model using grid search.

    Parameters
    ----------
    embeddings1, embeddings2 : np.ndarray
        Embedding matrices (n_pairs, embedding_dim)
    labels : np.ndarray
        Binary labels (1 for similar, 0 for dissimilar)
    metric : str, default='accuracy'
        Optimization metric: 'accuracy', 'f1', or 'correlation'

    Returns
    -------
    optimal_weight : float
        Optimal weight for cosine (0 to 1)
    results : dict
        Performance metrics at different weights

    Examples
    --------
    >>> optimal_w, results = optimize_hybrid_weights(emb1, emb2, labels)
    >>> print(f"Optimal cosine weight: {optimal_w:.2f}")
    """
    n_pairs = embeddings1.shape[0]

    # Precompute all cosine and xi values
    cosine_scores = np.array([
        cosine_similarity(embeddings1[i], embeddings2[i])
        for i in range(n_pairs)
    ])

    xi_scores = np.array([
        chatterjee_xi_symmetric(embeddings1[i], embeddings2[i])
        for i in range(n_pairs)
    ])

    def objective(weight_cosine):
        """Compute negative performance (for minimization)."""
        hybrid_scores = weight_cosine * cosine_scores + (1 - weight_cosine) * xi_scores

        if metric == 'accuracy':
            # Find optimal threshold
            threshold = np.median(hybrid_scores)
            predictions = (hybrid_scores >= threshold).astype(int)
            score = accuracy_score(labels, predictions)
        elif metric == 'f1':
            threshold = np.median(hybrid_scores)
            predictions = (hybrid_scores >= threshold).astype(int)
            score = f1_score(labels, predictions)
        elif metric == 'correlation':
            score, _ = spearmanr(hybrid_scores, labels)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return -score  # Negative for minimization

    # Grid search over weights
    weights = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.10, ..., 1.0
    scores = []

    for w in weights:
        score = -objective(w)  # Convert back to positive
        scores.append(score)

    # Find optimal weight
    optimal_idx = np.argmax(scores)
    optimal_weight = weights[optimal_idx]
    optimal_score = scores[optimal_idx]

    # Also get baseline scores
    cosine_only_score = scores[20]  # weight=1.0
    xi_only_score = scores[0]  # weight=0.0

    results = {
        'optimal_weight': optimal_weight,
        'optimal_score': optimal_score,
        'cosine_only_score': cosine_only_score,
        'xi_only_score': xi_only_score,
        'improvement_over_cosine': optimal_score - cosine_only_score,
        'improvement_over_xi': optimal_score - xi_only_score,
        'weights_tested': weights.tolist(),
        'scores_tested': scores,
        'metric': metric
    }

    return optimal_weight, results


def evaluate_hybrid_model(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    weights_to_test: Optional[np.ndarray] = None
) -> dict:
    """
    Comprehensive evaluation of hybrid model with different weight combinations.

    Parameters
    ----------
    embeddings1, embeddings2 : np.ndarray
        Embedding matrices
    labels : np.ndarray, optional
        Binary labels for classification evaluation
    scores : np.ndarray, optional
        Continuous scores for correlation evaluation
    weights_to_test : np.ndarray, optional
        Array of cosine weights to test. Default: [0.0, 0.1, ..., 1.0]

    Returns
    -------
    dict
        Comprehensive evaluation results
    """
    if weights_to_test is None:
        weights_to_test = np.linspace(0, 1, 11)

    n_pairs = embeddings1.shape[0]

    # Precompute base similarities
    cosine_scores = np.array([
        cosine_similarity(embeddings1[i], embeddings2[i])
        for i in range(n_pairs)
    ])

    xi_scores = np.array([
        chatterjee_xi_symmetric(embeddings1[i], embeddings2[i])
        for i in range(n_pairs)
    ])

    results = {
        'weights': weights_to_test.tolist(),
        'cosine_scores': cosine_scores,
        'xi_scores': xi_scores,
        'hybrid_scores': {}
    }

    # Evaluate each weight combination
    for weight in weights_to_test:
        hybrid_scores = weight * cosine_scores + (1 - weight) * xi_scores
        results['hybrid_scores'][f'w={weight:.1f}'] = hybrid_scores

        # Classification metrics
        if labels is not None:
            threshold = np.median(hybrid_scores)
            predictions = (hybrid_scores >= threshold).astype(int)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, zero_division=0)

            if 'classification' not in results:
                results['classification'] = {
                    'weights': [],
                    'accuracy': [],
                    'f1': []
                }

            results['classification']['weights'].append(weight)
            results['classification']['accuracy'].append(acc)
            results['classification']['f1'].append(f1)

        # Correlation metrics
        if scores is not None:
            corr, pval = spearmanr(hybrid_scores, scores)

            if 'correlation' not in results:
                results['correlation'] = {
                    'weights': [],
                    'spearman_r': [],
                    'spearman_p': []
                }

            results['correlation']['weights'].append(weight)
            results['correlation']['spearman_r'].append(corr)
            results['correlation']['spearman_p'].append(pval)

    # Find optimal weights
    if labels is not None:
        optimal_w_acc, opt_results_acc = optimize_hybrid_weights(
            embeddings1, embeddings2, labels, metric='accuracy'
        )
        results['optimal_accuracy'] = opt_results_acc

        optimal_w_f1, opt_results_f1 = optimize_hybrid_weights(
            embeddings1, embeddings2, labels, metric='f1'
        )
        results['optimal_f1'] = opt_results_f1

    if scores is not None:
        optimal_w_corr, opt_results_corr = optimize_hybrid_weights(
            embeddings1, embeddings2, scores, metric='correlation'
        )
        results['optimal_correlation'] = opt_results_corr

    return results


def adaptive_hybrid_similarity(
    x: np.ndarray,
    y: np.ndarray,
    nonlinearity_threshold: float = 0.3
) -> Tuple[float, str]:
    """
    Adaptive hybrid that automatically adjusts weights based on detected nonlinearity.

    Strategy:
    - If cosine and xi agree (high correlation), use cosine (faster)
    - If they disagree (suggests nonlinearity), weight xi more heavily

    Parameters
    ----------
    x, y : np.ndarray
        Input vectors
    nonlinearity_threshold : float
        Threshold for nonlinearity detection

    Returns
    -------
    similarity : float
        Adaptive hybrid similarity
    strategy : str
        Description of strategy used

    Note
    ----
    This is experimental and for demonstration purposes.
    """
    cos_sim = cosine_similarity(x, y)
    xi_sim = chatterjee_xi_symmetric(x, y)

    # Measure disagreement
    disagreement = abs(cos_sim - xi_sim)

    if disagreement < nonlinearity_threshold:
        # Low disagreement: use cosine (faster)
        return cos_sim, "linear (cosine only)"
    else:
        # High disagreement: weight xi more
        # Higher disagreement -> more weight on xi
        xi_weight = min(0.7, 0.3 + disagreement)
        cos_weight = 1 - xi_weight
        hybrid = cos_weight * cos_sim + xi_weight * xi_sim
        return hybrid, f"nonlinear (xi weight={xi_weight:.2f})"
