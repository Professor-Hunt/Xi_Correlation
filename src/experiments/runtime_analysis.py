"""
Runtime performance analysis for similarity metrics.

Compares computational cost of:
- Cosine similarity
- Chatterjee's Xi
- Symmetric Xi
- Hybrid models
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Callable
from pathlib import Path

from ..similarity.chatterjee_xi import chatterjee_xi, symmetric_xi
from ..similarity.metrics import cosine_similarity_score as cosine_similarity
try:
    from ..similarity.hybrid import hybrid_similarity
except ImportError:
    hybrid_similarity = None

# Alias for compatibility
chatterjee_xi_symmetric = symmetric_xi


def measure_runtime(
    func: Callable,
    x: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 100
) -> Dict[str, float]:
    """
    Measure runtime statistics for a similarity function.

    Parameters
    ----------
    func : callable
        Similarity function to measure
    x, y : np.ndarray
        Input vectors
    n_iterations : int
        Number of iterations for averaging

    Returns
    -------
    dict
        Runtime statistics (mean, std, min, max in milliseconds)
    """
    times = []

    # Warmup
    for _ in range(10):
        _ = func(x, y)

    # Actual measurement
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = func(x, y)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times)
    }


def runtime_vs_dimension(
    dimensions: List[int] = [10, 50, 100, 384, 768, 1024, 2048],
    n_iterations: int = 100,
    n_samples: int = 500
) -> pd.DataFrame:
    """
    Analyze runtime as a function of embedding dimension.

    Parameters
    ----------
    dimensions : list of int
        Embedding dimensions to test
    n_iterations : int
        Iterations per measurement
    n_samples : int
        Number of samples for Xi (for realistic scenarios)

    Returns
    -------
    pd.DataFrame
        Runtime results for each dimension
    """
    results = []

    metrics = {
        'cosine': cosine_similarity,
        'xi': chatterjee_xi,
        'xi_symmetric': chatterjee_xi_symmetric,
        'hybrid_0.5': lambda x, y: hybrid_similarity(x, y, weight_cosine=0.5)
    }

    for dim in dimensions:
        print(f"Testing dimension {dim}...")

        # Generate random vectors
        x = np.random.randn(dim)
        y = np.random.randn(dim)

        for metric_name, metric_func in metrics.items():
            stats = measure_runtime(metric_func, x, y, n_iterations)

            results.append({
                'dimension': dim,
                'metric': metric_name,
                **stats
            })

    return pd.DataFrame(results)


def runtime_vs_sample_size(
    sample_sizes: List[int] = [10, 50, 100, 500, 1000, 2000],
    dimension: int = 384,
    n_iterations: int = 50
) -> pd.DataFrame:
    """
    Analyze runtime as a function of sample size (for Xi).

    Note: This simulates the projection-based approach where we have
    multiple observations to compute Xi on.

    Parameters
    ----------
    sample_sizes : list of int
        Sample sizes to test
    dimension : int
        Embedding dimension
    n_iterations : int
        Iterations per measurement

    Returns
    -------
    pd.DataFrame
        Runtime results for each sample size
    """
    results = []

    for n_samples in sample_sizes:
        print(f"Testing sample size {n_samples}...")

        # Generate random data
        x = np.random.randn(n_samples)
        y = np.random.randn(n_samples)

        # Measure Xi computation
        stats = measure_runtime(chatterjee_xi, x, y, n_iterations)

        results.append({
            'sample_size': n_samples,
            'metric': 'xi',
            **stats
        })

        # For comparison, also measure on vectors of that "size"
        x_vec = np.random.randn(n_samples)
        y_vec = np.random.randn(n_samples)
        cos_stats = measure_runtime(cosine_similarity, x_vec, y_vec, n_iterations)

        results.append({
            'sample_size': n_samples,
            'metric': 'cosine',
            **cos_stats
        })

    return pd.DataFrame(results)


def runtime_comparison_table(
    dimension: int = 384,
    n_iterations: int = 200
) -> pd.DataFrame:
    """
    Create comprehensive runtime comparison table.

    Parameters
    ----------
    dimension : int
        Embedding dimension (default 384, typical for BERT)
    n_iterations : int
        Number of iterations for averaging

    Returns
    -------
    pd.DataFrame
        Comparison table with speedup factors
    """
    print(f"Runtime comparison for dimension {dimension}...")

    x = np.random.randn(dimension)
    y = np.random.randn(dimension)

    metrics = {
        'cosine': cosine_similarity,
        'xi': chatterjee_xi,
        'xi_symmetric': chatterjee_xi_symmetric,
        'hybrid_0.3': lambda x, y: hybrid_similarity(x, y, weight_cosine=0.3),
        'hybrid_0.5': lambda x, y: hybrid_similarity(x, y, weight_cosine=0.5),
        'hybrid_0.7': lambda x, y: hybrid_similarity(x, y, weight_cosine=0.7),
    }

    results = []
    baseline_time = None

    for metric_name, metric_func in metrics.items():
        stats = measure_runtime(metric_func, x, y, n_iterations)

        if baseline_time is None:
            baseline_time = stats['mean_ms']

        speedup = baseline_time / stats['mean_ms']

        results.append({
            'metric': metric_name,
            'mean_ms': stats['mean_ms'],
            'std_ms': stats['std_ms'],
            'speedup_vs_cosine': speedup if baseline_time == stats['mean_ms'] else baseline_time / stats['mean_ms']
        })

    df = pd.DataFrame(results)

    # Add relative slowdown compared to cosine
    cosine_time = df[df['metric'] == 'cosine']['mean_ms'].values[0]
    df['slowdown_vs_cosine'] = df['mean_ms'] / cosine_time

    return df


def pairwise_similarity_scaling(
    n_pairs_list: List[int] = [10, 50, 100, 500, 1000],
    dimension: int = 384
) -> pd.DataFrame:
    """
    Measure total time for computing pairwise similarities.

    This simulates real-world scenarios where you need to compute
    similarities for many pairs (e.g., ranking documents).

    Parameters
    ----------
    n_pairs_list : list of int
        Number of pairs to compute
    dimension : int
        Embedding dimension

    Returns
    -------
    pd.DataFrame
        Total runtime for different numbers of pairs
    """
    results = []

    metrics = {
        'cosine': cosine_similarity,
        'xi': chatterjee_xi,
        'xi_symmetric': chatterjee_xi_symmetric,
    }

    for n_pairs in n_pairs_list:
        print(f"Testing {n_pairs} pairs...")

        # Generate random pairs
        embeddings1 = np.random.randn(n_pairs, dimension)
        embeddings2 = np.random.randn(n_pairs, dimension)

        for metric_name, metric_func in metrics.items():
            start = time.perf_counter()

            for i in range(n_pairs):
                _ = metric_func(embeddings1[i], embeddings2[i])

            end = time.perf_counter()
            total_time_ms = (end - start) * 1000

            results.append({
                'n_pairs': n_pairs,
                'metric': metric_name,
                'total_ms': total_time_ms,
                'per_pair_ms': total_time_ms / n_pairs
            })

    return pd.DataFrame(results)


def run_comprehensive_runtime_analysis(
    save_dir: Path,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run all runtime analyses and save results.

    Parameters
    ----------
    save_dir : Path
        Directory to save results
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        All runtime analysis results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if verbose:
        print("=" * 60)
        print("RUNTIME ANALYSIS")
        print("=" * 60)

    # 1. Runtime vs dimension
    if verbose:
        print("\n1. Runtime vs Dimension")
        print("-" * 60)

    df_dim = runtime_vs_dimension(
        dimensions=[10, 50, 100, 384, 768, 1024],
        n_iterations=100
    )
    results['dimension_scaling'] = df_dim
    df_dim.to_csv(save_dir / 'runtime_vs_dimension.csv', index=False)

    if verbose:
        print("\nResults:")
        print(df_dim.pivot_table(
            values='mean_ms',
            index='dimension',
            columns='metric'
        ).round(4))

    # 2. Runtime vs sample size
    if verbose:
        print("\n2. Runtime vs Sample Size (for Xi)")
        print("-" * 60)

    df_samples = runtime_vs_sample_size(
        sample_sizes=[10, 50, 100, 500, 1000],
        dimension=384,
        n_iterations=50
    )
    results['sample_scaling'] = df_samples
    df_samples.to_csv(save_dir / 'runtime_vs_samples.csv', index=False)

    if verbose:
        print("\nResults:")
        print(df_samples.pivot_table(
            values='mean_ms',
            index='sample_size',
            columns='metric'
        ).round(4))

    # 3. Comprehensive comparison
    if verbose:
        print("\n3. Comprehensive Comparison (d=384)")
        print("-" * 60)

    df_comparison = runtime_comparison_table(dimension=384, n_iterations=200)
    results['comparison'] = df_comparison
    df_comparison.to_csv(save_dir / 'runtime_comparison.csv', index=False)

    if verbose:
        print("\nResults:")
        print(df_comparison.to_string(index=False))

    # 4. Pairwise similarity scaling
    if verbose:
        print("\n4. Pairwise Similarity Scaling")
        print("-" * 60)

    df_pairwise = pairwise_similarity_scaling(
        n_pairs_list=[10, 50, 100, 500],
        dimension=384
    )
    results['pairwise_scaling'] = df_pairwise
    df_pairwise.to_csv(save_dir / 'runtime_pairwise.csv', index=False)

    if verbose:
        print("\nResults:")
        print(df_pairwise.pivot_table(
            values='total_ms',
            index='n_pairs',
            columns='metric'
        ).round(2))

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Results saved to {save_dir}")
        print('=' * 60)

    return results


if __name__ == "__main__":
    import sys
    save_dir = Path("results/runtime") if len(sys.argv) == 1 else Path(sys.argv[1])
    results = run_comprehensive_runtime_analysis(save_dir, verbose=True)
