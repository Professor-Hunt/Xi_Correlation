"""
Statistical testing utilities for similarity metrics.

Provides bootstrap confidence intervals, permutation tests,
and significance testing for comparing correlation measures.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List, Dict
from scipy import stats
import warnings


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Input data (can be 1D or 2D)
    statistic : callable
        Function that computes the statistic from data
        Should take data as input and return a scalar
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    point_estimate : float
        Point estimate of the statistic
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval

    Examples
    --------
    >>> data = np.random.randn(100)
    >>> mean_est, ci_low, ci_high = bootstrap_confidence_interval(
    ...     data, np.mean, n_bootstrap=1000
    ... )
    """
    rng = np.random.RandomState(random_state)
    n = len(data)

    # Compute point estimate
    point_estimate = statistic(data)

    # Bootstrap sampling
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.randint(0, n, size=n)
        bootstrap_sample = data[indices]
        bootstrap_estimates[i] = statistic(bootstrap_sample)

    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return float(point_estimate), float(ci_lower), float(ci_upper)


def bootstrap_confidence_interval_2sample(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a two-sample statistic.

    Parameters
    ----------
    x : np.ndarray
        First sample
    y : np.ndarray
        Second sample
    statistic : callable
        Function that takes (x, y) and returns a scalar
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level
    random_state : int, optional
        Random seed

    Returns
    -------
    point_estimate : float
        Point estimate
    ci_lower : float
        Lower CI bound
    ci_upper : float
        Upper CI bound

    Examples
    --------
    >>> from src.similarity.chatterjee_xi import chatterjee_xi
    >>> x = np.random.randn(50)
    >>> y = x + np.random.randn(50) * 0.1  # Correlated
    >>> est, low, high = bootstrap_confidence_interval_2sample(
    ...     x, y, chatterjee_xi, n_bootstrap=1000
    ... )
    """
    rng = np.random.RandomState(random_state)

    # Compute point estimate
    point_estimate = statistic(x, y)

    # Bootstrap sampling
    n = len(x)
    bootstrap_estimates = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample pairs with replacement
        indices = rng.randint(0, n, size=n)
        x_boot = x[indices]
        y_boot = y[indices]
        bootstrap_estimates[i] = statistic(x_boot, y_boot)

    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return float(point_estimate), float(ci_lower), float(ci_upper)


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable,
    n_permutations: int = 10000,
    alternative: str = 'two-sided',
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Perform permutation test for independence between x and y.

    Tests the null hypothesis that x and y are independent by randomly
    permuting the pairing between x and y values.

    Parameters
    ----------
    x : np.ndarray
        First variable (1D array)
    y : np.ndarray
        Second variable (1D array)
    statistic : callable
        Function that computes test statistic from (x, y)
    n_permutations : int, default=10000
        Number of random permutations
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'greater', or 'less'
    random_state : int, optional
        Random seed

    Returns
    -------
    observed_statistic : float
        Observed test statistic
    p_value : float
        Permutation-based p-value

    Examples
    --------
    >>> from src.similarity.chatterjee_xi import chatterjee_xi
    >>> x = np.random.randn(50)
    >>> y = x ** 2  # Dependent
    >>> obs_stat, p_val = permutation_test(x, y, chatterjee_xi)
    >>> print(f"p-value: {p_val:.4f}")
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    rng = np.random.RandomState(random_state)

    # Compute observed statistic
    observed = statistic(x, y)

    # Permutation distribution
    perm_statistics = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Permute y
        y_perm = rng.permutation(y)
        perm_statistics[i] = statistic(x, y_perm)

    # Compute p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_statistics) >= np.abs(observed))
    elif alternative == 'greater':
        p_value = np.mean(perm_statistics >= observed)
    elif alternative == 'less':
        p_value = np.mean(perm_statistics <= observed)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    return float(observed), float(p_value)


def compare_metrics_significance(
    x: np.ndarray,
    y: np.ndarray,
    metric1: Callable,
    metric2: Callable,
    n_bootstrap: int = 10000,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """
    Compare two metrics and test if their difference is significant.

    Uses bootstrap to estimate confidence interval for the difference.

    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable
    metric1 : callable
        First metric function
    metric2 : callable
        Second metric function
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with keys:
        - 'metric1_value': Value of first metric
        - 'metric2_value': Value of second metric
        - 'difference': metric1 - metric2
        - 'ci_lower': Lower CI bound for difference
        - 'ci_upper': Upper CI bound for difference
        - 'p_value': Bootstrap p-value (proportion of samples where difference <= 0)

    Examples
    --------
    >>> from src.similarity.chatterjee_xi import chatterjee_xi
    >>> from src.similarity.metrics import cosine_similarity_score
    >>> x = np.random.randn(100)
    >>> y = x ** 2
    >>> results = compare_metrics_significance(x, y, chatterjee_xi, cosine_similarity_score)
    """
    rng = np.random.RandomState(random_state)
    n = len(x)

    # Observed values
    metric1_value = metric1(x, y)
    metric2_value = metric2(x, y)
    observed_diff = metric1_value - metric2_value

    # Bootstrap
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        x_boot = x[indices]
        y_boot = y[indices]

        m1 = metric1(x_boot, y_boot)
        m2 = metric2(x_boot, y_boot)
        bootstrap_diffs[i] = m1 - m2

    # Confidence interval for difference
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    # P-value (proportion of bootstrap samples where difference has opposite sign)
    if observed_diff > 0:
        p_value = np.mean(bootstrap_diffs <= 0)
    else:
        p_value = np.mean(bootstrap_diffs >= 0)

    return {
        'metric1_value': float(metric1_value),
        'metric2_value': float(metric2_value),
        'difference': float(observed_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value)
    }


def effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for two groups.

    Parameters
    ----------
    group1 : np.ndarray
        First group of values
    group2 : np.ndarray
        Second group of values

    Returns
    -------
    float
        Cohen's d (standardized mean difference)

    Notes
    -----
    Effect size interpretation:
    - Small: d = 0.2
    - Medium: d = 0.5
    - Large: d = 0.8
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean1 - mean2) / pooled_std
    return float(d)


def mann_whitney_u_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test (non-parametric test for difference in distributions).

    Parameters
    ----------
    group1 : np.ndarray
        First group
    group2 : np.ndarray
        Second group
    alternative : str, default='two-sided'
        Alternative hypothesis

    Returns
    -------
    statistic : float
        U statistic
    p_value : float
        P-value
    """
    statistic, p_value = stats.mannwhitneyu(
        group1, group2,
        alternative=alternative
    )
    return float(statistic), float(p_value)


def mcnemar_test(
    metric1_correct: np.ndarray,
    metric2_correct: np.ndarray
) -> Tuple[float, float]:
    """
    McNemar's test for comparing classification performance of two metrics.

    Parameters
    ----------
    metric1_correct : np.ndarray
        Binary array indicating correct classifications for metric 1
    metric2_correct : np.ndarray
        Binary array indicating correct classifications for metric 2

    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value

    Examples
    --------
    >>> # Metric 1 correct on samples 0, 1, 2
    >>> # Metric 2 correct on samples 0, 2, 3
    >>> metric1 = np.array([1, 1, 1, 0])
    >>> metric2 = np.array([1, 0, 1, 1])
    >>> stat, p_val = mcnemar_test(metric1, metric2)
    """
    # Create contingency table
    # Rows: metric1 (wrong, correct)
    # Cols: metric2 (wrong, correct)
    n_00 = np.sum((metric1_correct == 0) & (metric2_correct == 0))
    n_01 = np.sum((metric1_correct == 0) & (metric2_correct == 1))
    n_10 = np.sum((metric1_correct == 1) & (metric2_correct == 0))
    n_11 = np.sum((metric1_correct == 1) & (metric2_correct == 1))

    contingency = np.array([[n_00, n_01], [n_10, n_11]])

    # McNemar's test focuses on discordant pairs (n_01, n_10)
    # Under null hypothesis: n_01 and n_10 should be equal
    if n_01 + n_10 == 0:
        return 0.0, 1.0

    # Use continuity correction
    statistic = (np.abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return float(statistic), float(p_value)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute classification metrics (accuracy, precision, recall, F1).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Predicted scores
    threshold : float, optional
        Classification threshold. If None, uses 0 or optimal threshold.

    Returns
    -------
    dict
        Dictionary with classification metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    if threshold is None:
        # Find optimal threshold
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]

    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_scores))
    except:
        metrics['auc'] = np.nan

    return metrics


def friedman_test(*groups) -> Tuple[float, float]:
    """
    Friedman test for comparing multiple related samples.

    Non-parametric alternative to repeated measures ANOVA.

    Parameters
    ----------
    *groups : array_like
        Two or more related sample arrays

    Returns
    -------
    statistic : float
        Friedman test statistic
    p_value : float
        P-value

    Examples
    --------
    >>> # Compare 3 metrics on same dataset
    >>> metric1_scores = np.random.rand(20)
    >>> metric2_scores = np.random.rand(20)
    >>> metric3_scores = np.random.rand(20)
    >>> stat, p_val = friedman_test(metric1_scores, metric2_scores, metric3_scores)
    """
    statistic, p_value = stats.friedmanchisquare(*groups)
    return float(statistic), float(p_value)


def wilcoxon_signed_rank_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples.

    Non-parametric test for comparing two related samples.

    Parameters
    ----------
    x : np.ndarray
        First sample
    y : np.ndarray
        Second sample
    alternative : str, default='two-sided'
        Alternative hypothesis

    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value
    """
    statistic, p_value = stats.wilcoxon(x, y, alternative=alternative)
    return float(statistic), float(p_value)


class StatisticalComparison:
    """
    Comprehensive statistical comparison of similarity metrics.

    Examples
    --------
    >>> comparison = StatisticalComparison()
    >>> x = np.random.randn(100)
    >>> y = x ** 2
    >>> from src.similarity.chatterjee_xi import chatterjee_xi
    >>> from src.similarity.metrics import cosine_similarity_score
    >>> results = comparison.compare_two_metrics(
    ...     x, y, chatterjee_xi, cosine_similarity_score
    ... )
    """

    def __init__(self, n_bootstrap: int = 10000, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def compare_two_metrics(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metric1: Callable,
        metric2: Callable
    ) -> Dict:
        """Comprehensive comparison of two metrics."""
        results = compare_metrics_significance(
            x, y, metric1, metric2,
            n_bootstrap=self.n_bootstrap,
            random_state=self.random_state
        )

        # Add permutation tests
        _, p1 = permutation_test(x, y, metric1, random_state=self.random_state)
        _, p2 = permutation_test(x, y, metric2, random_state=self.random_state)

        results['metric1_permutation_pvalue'] = p1
        results['metric2_permutation_pvalue'] = p2

        return results

    def compare_on_multiple_pairs(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray]],
        metric1: Callable,
        metric2: Callable
    ) -> Dict:
        """Compare two metrics across multiple data pairs."""
        metric1_scores = []
        metric2_scores = []

        for x, y in pairs:
            metric1_scores.append(metric1(x, y))
            metric2_scores.append(metric2(x, y))

        metric1_scores = np.array(metric1_scores)
        metric2_scores = np.array(metric2_scores)

        # Wilcoxon signed-rank test
        stat, p_value = wilcoxon_signed_rank_test(metric1_scores, metric2_scores)

        # Effect size
        effect_size = effect_size_cohens_d(metric1_scores, metric2_scores)

        return {
            'metric1_mean': float(np.mean(metric1_scores)),
            'metric1_std': float(np.std(metric1_scores)),
            'metric2_mean': float(np.mean(metric2_scores)),
            'metric2_std': float(np.std(metric2_scores)),
            'wilcoxon_statistic': stat,
            'wilcoxon_pvalue': p_value,
            'effect_size_cohens_d': effect_size,
            'n_pairs': len(pairs)
        }
