"""
Synthetic experiments to test similarity metrics on known functional relationships.

Generates data with specific properties (linear, nonlinear, independent) and
compares how different metrics capture these relationships.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
from pathlib import Path
import json

from ..similarity.chatterjee_xi import chatterjee_xi, symmetric_xi
from ..similarity.metrics import compute_all_similarities
from ..utils.statistics import bootstrap_confidence_interval_2sample


def generate_synthetic_data(
    n_samples: int = 500,
    n_features: int = 1,
    relationship: str = 'linear',
    noise_level: float = 0.05,
    random_state: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with a specific relationship.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features (for high-dimensional data)
    relationship : str
        Type of relationship:
        - 'linear': y = x + noise
        - 'quadratic': y = x^2
        - 'cubic': y = x^3
        - 'absolute': y = |x|
        - 'sine': y = sin(x)
        - 'cosine': y = cos(x)
        - 'exponential': y = exp(x/10)
        - 'logarithmic': y = log(|x| + 1)
        - 'inverse': y = 1/(x + eps)
        - 'independent': y = random
    noise_level : float
        Standard deviation of Gaussian noise to add
    random_state : int, optional
        Random seed

    Returns
    -------
    x : np.ndarray
        Independent variable(s), shape (n_samples,) or (n_samples, n_features)
    y : np.ndarray
        Dependent variable, shape (n_samples,) or (n_samples, n_features)
    """
    rng = np.random.RandomState(random_state)

    if n_features == 1:
        # 1D case
        x = rng.randn(n_samples)

        if relationship == 'linear':
            y = x + noise_level * rng.randn(n_samples)
        elif relationship == 'quadratic':
            y = x ** 2
        elif relationship == 'cubic':
            y = x ** 3
        elif relationship == 'absolute':
            y = np.abs(x)
        elif relationship == 'sine':
            y = np.sin(2 * np.pi * x)
        elif relationship == 'cosine':
            y = np.cos(2 * np.pi * x)
        elif relationship == 'exponential':
            y = np.exp(x / 10)
        elif relationship == 'logarithmic':
            y = np.log(np.abs(x) + 1)
        elif relationship == 'inverse':
            y = 1 / (np.abs(x) + 0.1)
        elif relationship == 'independent':
            y = rng.randn(n_samples)
        else:
            raise ValueError(f"Unknown relationship: {relationship}")

    else:
        # High-dimensional case
        x = rng.randn(n_samples, n_features)

        if relationship == 'linear':
            # y = x + noise
            y = x + noise_level * rng.randn(n_samples, n_features)
        elif relationship == 'quadratic':
            y = x ** 2
        elif relationship == 'absolute':
            y = np.abs(x)
        elif relationship == 'independent':
            y = rng.randn(n_samples, n_features)
        else:
            # For other relationships, apply elementwise
            if relationship == 'cubic':
                y = x ** 3
            elif relationship == 'sine':
                y = np.sin(2 * np.pi * x)
            elif relationship == 'exponential':
                y = np.exp(x / 10)
            else:
                raise ValueError(f"Relationship {relationship} not supported for high-dimensional data")

    return x, y


def run_single_synthetic_experiment(
    relationship: str,
    n_samples: int = 500,
    n_features: int = 1,
    n_repetitions: int = 10,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Run synthetic experiment for a single relationship type.

    Parameters
    ----------
    relationship : str
        Type of relationship
    n_samples : int
        Number of samples per repetition
    n_features : int
        Dimensionality of data
    n_repetitions : int
        Number of repetitions
    random_state : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns for each metric
    """
    results = []
    base_seed = random_state if random_state is not None else 42

    for rep in range(n_repetitions):
        seed = base_seed + rep
        x, y = generate_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            relationship=relationship,
            random_state=seed
        )

        # Flatten if needed
        if x.ndim > 1:
            x = x.reshape(-1)
            y = y.reshape(-1)

        # Compute all metrics
        metrics = compute_all_similarities(x, y)
        metrics['relationship'] = relationship
        metrics['repetition'] = rep
        metrics['n_samples'] = n_samples

        results.append(metrics)

    return pd.DataFrame(results)


def run_synthetic_experiments(
    relationships: Optional[List[str]] = None,
    n_samples: int = 500,
    n_features: int = 1,
    n_repetitions: int = 10,
    random_state: int = 42,
    save_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive synthetic experiments.

    Parameters
    ----------
    relationships : list of str, optional
        Relationships to test. If None, tests all standard relationships.
    n_samples : int
        Number of samples per experiment
    n_features : int
        Dimensionality
    n_repetitions : int
        Number of repetitions per relationship
    random_state : int
        Random seed
    save_dir : Path, optional
        Directory to save results

    Returns
    -------
    dict
        Dictionary mapping relationship names to result DataFrames

    Examples
    --------
    >>> results = run_synthetic_experiments(n_repetitions=5)
    >>> print(results['linear']['xi'].mean())
    """
    if relationships is None:
        relationships = [
            'linear', 'quadratic', 'cubic', 'absolute',
            'sine', 'exponential', 'independent'
        ]

    all_results = {}

    print("Running synthetic experiments...")
    for relationship in relationships:
        print(f"  Testing {relationship} relationship...")
        df = run_single_synthetic_experiment(
            relationship=relationship,
            n_samples=n_samples,
            n_features=n_features,
            n_repetitions=n_repetitions,
            random_state=random_state
        )
        all_results[relationship] = df

    # Create summary
    summary_rows = []
    for relationship, df in all_results.items():
        summary_rows.append({
            'relationship': relationship,
            'cosine_mean': df['cosine'].mean(),
            'cosine_std': df['cosine'].std(),
            'xi_mean': df['xi'].mean(),
            'xi_std': df['xi'].std(),
            'pearson_mean': df['pearson'].mean(),
            'pearson_std': df['pearson'].std(),
            'spearman_mean': df['spearman'].mean(),
            'spearman_std': df['spearman'].std(),
        })

    summary_df = pd.DataFrame(summary_rows)
    all_results['summary'] = summary_df

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, df in all_results.items():
            csv_path = save_dir / f'synthetic_{name}.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved {name} results to {csv_path}")

        # Save summary statistics
        summary_path = save_dir / 'synthetic_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_rows, f, indent=2)

    print("\nSynthetic experiments complete!")
    return all_results


def run_dimensionality_experiment(
    dimensions: List[int] = [1, 10, 50, 100, 384, 768],
    n_samples: int = 100,
    n_repetitions: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Test how metrics perform across different dimensionalities.

    Parameters
    ----------
    dimensions : list of int
        Dimensionalities to test
    n_samples : int
        Number of samples per experiment
    n_repetitions : int
        Repetitions per dimension
    random_state : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Results with metrics for each dimensionality
    """
    results = []

    print("Running dimensionality experiments...")
    for dim in dimensions:
        print(f"  Testing dimension {dim}...")
        for rep in range(n_repetitions):
            seed = random_state + rep

            # Test linear relationship
            x, y = generate_synthetic_data(
                n_samples=n_samples,
                n_features=dim,
                relationship='linear',
                random_state=seed
            )

            # Flatten
            x = x.reshape(-1)
            y = y.reshape(-1)

            metrics = compute_all_similarities(x, y)
            metrics['dimension'] = dim
            metrics['repetition'] = rep

            results.append(metrics)

    return pd.DataFrame(results)


def run_noise_robustness_experiment(
    noise_levels: List[float] = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
    n_samples: int = 500,
    n_repetitions: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Test robustness to noise.

    Parameters
    ----------
    noise_levels : list of float
        Noise levels to test
    n_samples : int
        Number of samples
    n_repetitions : int
        Repetitions per noise level
    random_state : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Results showing metric performance at different noise levels
    """
    results = []

    print("Running noise robustness experiments...")
    for noise in noise_levels:
        print(f"  Testing noise level {noise}...")
        for rep in range(n_repetitions):
            seed = random_state + rep
            rng = np.random.RandomState(seed)

            # Generate linear relationship
            x = rng.randn(n_samples)
            y = x + noise * rng.randn(n_samples)

            metrics = compute_all_similarities(x, y)
            metrics['noise_level'] = noise
            metrics['repetition'] = rep

            results.append(metrics)

    return pd.DataFrame(results)


def visualize_synthetic_relationships(
    n_samples: int = 200,
    random_state: int = 42,
    save_path: Optional[Path] = None
):
    """
    Generate and visualize synthetic relationships for paper figures.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    from ..utils.visualization import plot_correlation_comparison

    relationships = ['linear', 'quadratic', 'absolute', 'sine', 'independent']

    # Generate data
    y_dict = {}
    metrics_dict = {}

    for rel in relationships:
        x, y = generate_synthetic_data(
            n_samples=n_samples,
            relationship=rel,
            random_state=random_state
        )

        # Sort for better visualization
        if rel != 'independent':
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
        else:
            x_sorted = x
            y_sorted = y

        y_dict[rel.capitalize()] = y_sorted

        # Compute metrics
        metrics = compute_all_similarities(x, y)
        metrics_dict[rel.capitalize()] = {
            'Cosine': metrics['cosine'],
            'Xi': metrics['xi'],
            'Pearson': metrics['pearson'],
            'Spearman': metrics['spearman']
        }

    # Use same x for visualization
    x, _ = generate_synthetic_data(n_samples=n_samples, relationship='linear', random_state=random_state)
    x_sorted = np.sort(x)

    # Create plot
    fig = plot_correlation_comparison(
        x_sorted,
        y_dict,
        metrics=metrics_dict,
        title="Comparison of Correlation Measures Across Functional Relationships",
        save_path=save_path,
        figsize=(14, 10)
    )

    return fig
