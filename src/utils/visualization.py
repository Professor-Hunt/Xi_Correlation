"""
Visualization utilities for similarity metrics and correlation analysis.

Provides publication-quality plotting functions for research papers.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_correlation_comparison(
    x: np.ndarray,
    y_dict: Dict[str, np.ndarray],
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    title: str = "Correlation Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    """
    Plot multiple functional relationships and their correlation metrics.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (1D array)
    y_dict : dict
        Dictionary mapping relationship names to y arrays
        e.g., {'Linear': y_linear, 'Quadratic': y_quad}
    metrics : dict, optional
        Dictionary of {relationship_name: {metric_name: value}}
        If provided, adds text annotations with metric values
    title : str
        Plot title
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y_dict = {
    ...     'Linear': x,
    ...     'Quadratic': x**2,
    ...     'Cubic': x**3
    ... }
    >>> fig = plot_correlation_comparison(x, y_dict)
    """
    n_plots = len(y_dict)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_plots > 2 else list(axes)

    for idx, (name, y) in enumerate(y_dict.items()):
        ax = axes[idx]
        ax.scatter(x, y, alpha=0.6, s=20, edgecolors='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{name} Relationship')
        ax.grid(True, alpha=0.3)

        # Add metrics text if provided
        if metrics and name in metrics:
            text_parts = []
            for metric_name, value in metrics[name].items():
                text_parts.append(f'{metric_name}: {value:.3f}')
            text = '\n'.join(text_parts)
            ax.text(0.05, 0.95, text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Similarity Matrix",
    cmap: str = 'RdYlGn',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 8),
    annot: bool = True,
    fmt: str = '.2f'
) -> plt.Figure:
    """
    Plot a similarity matrix as a heatmap.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square matrix of similarities
    labels : list of str, optional
        Labels for rows/columns
    title : str
        Plot title
    cmap : str
        Colormap name
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    annot : bool
        Whether to annotate cells with values
    fmt : str
        Format string for annotations

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        similarity_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'label': 'Similarity'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_metric_comparison(
    results_df: pd.DataFrame,
    x_metric: str = 'cosine',
    y_metric: str = 'xi',
    hue: Optional[str] = 'label',
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6)
) -> plt.Figure:
    """
    Create a scatter plot comparing two similarity metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns for different metrics
    x_metric : str
        Column name for x-axis metric
    y_metric : str
        Column name for y-axis metric
    hue : str, optional
        Column name for color coding
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hue and hue in results_df.columns:
        for label in results_df[hue].unique():
            mask = results_df[hue] == label
            ax.scatter(
                results_df.loc[mask, x_metric],
                results_df.loc[mask, y_metric],
                label=f'{hue}={label}',
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        ax.legend()
    else:
        ax.scatter(
            results_df[x_metric],
            results_df[y_metric],
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel(f'{x_metric.capitalize()} Similarity')
    ax.set_ylabel(f'{y_metric.capitalize()} Similarity')

    if title is None:
        title = f'{x_metric.capitalize()} vs {y_metric.capitalize()}'
    ax.set_title(title, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    title: str = "Metric Performance Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Create a bar plot comparing performance across metrics.

    Parameters
    ----------
    results : dict
        Nested dictionary: {metric_name: {measure: value}}
        e.g., {'cosine': {'accuracy': 0.85, 'f1': 0.82}, ...}
    title : str
        Plot title
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    df = pd.DataFrame(results).T
    df.index.name = 'Metric'

    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind='bar', ax=ax, rot=45)

    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend(title='Measure')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_distribution_comparison(
    data_dict: Dict[str, np.ndarray],
    title: str = "Distribution Comparison",
    xlabel: str = "Value",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Plot overlapping distributions for multiple datasets.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping labels to data arrays
    title : str
        Plot title
    xlabel : str
        X-axis label
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, data in data_dict.items():
        ax.hist(data, bins=30, alpha=0.5, label=label, edgecolor='black')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: Dict[str, np.ndarray],
    title: str = "ROC Curves",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curves for multiple metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : dict
        Dictionary mapping metric names to predicted scores
    title : str
        Plot title
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)

    for metric_name, scores in y_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{metric_name} (AUC = {roc_auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def create_results_table(
    results_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    latex: bool = True
) -> str:
    """
    Create a formatted table from results DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results data
    save_path : str or Path, optional
        Path to save table (as .tex or .csv)
    latex : bool
        Whether to format as LaTeX table

    Returns
    -------
    str
        Formatted table string
    """
    if latex:
        table_str = results_df.to_latex(
            index=False,
            float_format='%.3f',
            escape=False,
            column_format='l' + 'c' * (len(results_df.columns) - 1)
        )
    else:
        table_str = results_df.to_string(index=False)

    if save_path:
        save_path = Path(save_path)
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"Table saved to {save_path}")

    return table_str


def plot_synthetic_experiments(
    results: Dict[str, pd.DataFrame],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (14, 10)
) -> plt.Figure:
    """
    Create comprehensive visualization of synthetic experiments.

    Parameters
    ----------
    results : dict
        Dictionary mapping experiment names to results DataFrames
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Subplot configurations
    subplots = [
        ('Linear', 0, 0),
        ('Quadratic', 0, 1),
        ('Absolute', 1, 0),
        ('Random', 1, 1),
    ]

    for name, row, col in subplots:
        if name.lower() in results:
            ax = fig.add_subplot(gs[row, col])
            df = results[name.lower()]

            # Bar plot of metrics
            metrics = ['cosine', 'xi', 'pearson', 'spearman']
            values = [df[m].mean() if m in df.columns else 0 for m in metrics]

            ax.bar(metrics, values, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name} Relationship')
            ax.set_ylabel('Correlation')
            ax.set_ylim([-0.2, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)

    # Summary subplot
    ax_summary = fig.add_subplot(gs[2, :])
    summary_data = []
    for name in ['linear', 'quadratic', 'absolute', 'random']:
        if name in results:
            df = results[name]
            summary_data.append({
                'Relationship': name.capitalize(),
                'Cosine': df['cosine'].mean() if 'cosine' in df.columns else 0,
                'Xi': df['xi'].mean() if 'xi' in df.columns else 0,
                'Pearson': df['pearson'].mean() if 'pearson' in df.columns else 0,
                'Spearman': df['spearman'].mean() if 'spearman' in df.columns else 0,
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        x = np.arange(len(summary_df))
        width = 0.2

        for i, metric in enumerate(['Cosine', 'Xi', 'Pearson', 'Spearman']):
            ax_summary.bar(x + i * width, summary_df[metric], width, label=metric)

        ax_summary.set_xlabel('Relationship Type')
        ax_summary.set_ylabel('Mean Correlation')
        ax_summary.set_title('Summary of All Relationships')
        ax_summary.set_xticks(x + width * 1.5)
        ax_summary.set_xticklabels(summary_df['Relationship'])
        ax_summary.legend()
        ax_summary.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Synthetic Correlation Experiments', fontsize=16, fontweight='bold')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig
