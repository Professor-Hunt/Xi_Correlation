"""
Benchmark evaluation on standard semantic similarity datasets.

Supports:
- STS-B (Semantic Textual Similarity Benchmark)
- SICK (Sentences Involving Compositional Knowledge)
- Custom sentence pair datasets
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from scipy.stats import spearmanr, pearsonr

from ..similarity.embeddings import EmbeddingModel
from ..similarity.metrics import compute_all_similarities
from ..utils.statistics import compute_classification_metrics


# Sample sentence pairs for quick testing
DEFAULT_SENTENCE_PAIRS = [
    ("The quick brown fox jumps over the lazy dog.",
     "A swift auburn fox leaps above a sleepy canine.", 1),
    ("A man is playing guitar on stage.",
     "Someone is strumming a musical instrument in front of an audience.", 1),
    ("The capital of France is Paris.",
     "Paris is the capital city of France.", 1),
    ("Ice cream tastes delicious on a hot day.",
     "Eating frozen dessert is enjoyable when it's warm outside.", 1),
    ("The stock market crashed causing panic.",
     "An octopus is swimming in the ocean.", 0),
    ("A student is studying mathematics.",
     "Fish live in the coral reef.", 0),
    ("She went shopping for a new dress.",
     "The earth revolves around the sun.", 0),
    ("He is writing code in Python.",
     "The flowers bloom in spring.", 0),
    # Additional pairs for better coverage
    ("A dog is running in the park.",
     "A canine is jogging through the garden.", 1),
    ("The weather is sunny today.",
     "It's a bright and clear day.", 1),
    ("She enjoys reading books.",
     "He prefers watching television.", 0),
    ("The computer is broken.",
     "The sky is blue.", 0),
]


class BenchmarkDataset:
    """
    Container for benchmark datasets.

    Parameters
    ----------
    sentences1 : list of str
        First sentences
    sentences2 : list of str
        Second sentences
    labels : np.ndarray, optional
        Binary labels (1 for similar, 0 for dissimilar)
    scores : np.ndarray, optional
        Continuous similarity scores (e.g., human ratings)
    name : str
        Dataset name
    """

    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        labels: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        name: str = "custom"
    ):
        assert len(sentences1) == len(sentences2), "Sentence lists must have equal length"

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.scores = scores
        self.name = name

    def __len__(self) -> int:
        return len(self.sentences1)

    def __repr__(self) -> str:
        return f"BenchmarkDataset(name='{self.name}', n_pairs={len(self)})"

    @classmethod
    def from_tuples(cls, pairs: List[Tuple[str, str, int]], name: str = "custom"):
        """Create dataset from list of (sent1, sent2, label) tuples."""
        sentences1 = [p[0] for p in pairs]
        sentences2 = [p[1] for p in pairs]
        labels = np.array([p[2] for p in pairs])
        return cls(sentences1, sentences2, labels=labels, name=name)

    @classmethod
    def load_default(cls):
        """Load default test dataset."""
        return cls.from_tuples(DEFAULT_SENTENCE_PAIRS, name="default")


def evaluate_on_dataset(
    dataset: BenchmarkDataset,
    model: EmbeddingModel,
    metrics: List[str] = ['cosine', 'xi', 'xi_symmetric'],
    verbose: bool = True
) -> Dict:
    """
    Evaluate similarity metrics on a benchmark dataset.

    Parameters
    ----------
    dataset : BenchmarkDataset
        Dataset to evaluate on
    model : EmbeddingModel
        Embedding model to use
    metrics : list of str
        Metrics to evaluate
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Evaluation results including:
        - 'embeddings': Computed embeddings
        - 'similarities': Similarity scores for each metric
        - 'classification_metrics': Accuracy, F1, etc. (if labels available)
        - 'correlation_metrics': Correlation with human scores (if scores available)
    """
    if verbose:
        print(f"Evaluating on {dataset.name} dataset ({len(dataset)} pairs)...")

    # Encode sentences
    if verbose:
        print("  Computing embeddings...")

    all_sentences = dataset.sentences1 + dataset.sentences2
    all_embeddings = model.encode(all_sentences, show_progress_bar=verbose)

    # Split embeddings
    n = len(dataset)
    embeddings1 = all_embeddings[:n]
    embeddings2 = all_embeddings[n:]

    # Compute similarities
    if verbose:
        print("  Computing similarities...")

    similarities = {metric: [] for metric in metrics}

    for i in range(n):
        emb1 = embeddings1[i]
        emb2 = embeddings2[i]

        # Compute all metrics for this pair
        all_sims = compute_all_similarities(emb1, emb2)

        for metric in metrics:
            if metric in all_sims:
                similarities[metric].append(all_sims[metric])

    # Convert to arrays
    for metric in metrics:
        similarities[metric] = np.array(similarities[metric])

    results = {
        'dataset_name': dataset.name,
        'n_pairs': n,
        'model_name': model.model_name,
        'similarities': similarities
    }

    # Classification metrics (if labels available)
    if dataset.labels is not None:
        if verbose:
            print("  Computing classification metrics...")

        classification_results = {}
        for metric in metrics:
            if metric in similarities:
                clf_metrics = compute_classification_metrics(
                    dataset.labels,
                    similarities[metric]
                )
                classification_results[metric] = clf_metrics

        results['classification_metrics'] = classification_results

    # Correlation metrics (if continuous scores available)
    if dataset.scores is not None:
        if verbose:
            print("  Computing correlation metrics...")

        correlation_results = {}
        for metric in metrics:
            if metric in similarities:
                pearson_r, pearson_p = pearsonr(dataset.scores, similarities[metric])
                spearman_r, spearman_p = spearmanr(dataset.scores, similarities[metric])

                correlation_results[metric] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }

        results['correlation_metrics'] = correlation_results

    if verbose:
        print("  Done!")

    return results


def run_benchmark_evaluation(
    datasets: Optional[List[BenchmarkDataset]] = None,
    model_names: List[str] = ['all-MiniLM-L6-v2'],
    metrics: List[str] = ['cosine', 'xi', 'xi_symmetric'],
    save_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict:
    """
    Run comprehensive benchmark evaluation.

    Parameters
    ----------
    datasets : list of BenchmarkDataset, optional
        Datasets to evaluate. If None, uses default dataset.
    model_names : list of str
        Embedding models to evaluate
    metrics : list of str
        Similarity metrics to evaluate
    save_dir : Path, optional
        Directory to save results
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Comprehensive results for all combinations of datasets and models

    Examples
    --------
    >>> dataset = BenchmarkDataset.load_default()
    >>> results = run_benchmark_evaluation(
    ...     datasets=[dataset],
    ...     model_names=['all-MiniLM-L6-v2']
    ... )
    """
    if datasets is None:
        datasets = [BenchmarkDataset.load_default()]

    all_results = []

    for model_name in model_names:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Loading model: {model_name}")
            print('='*60)

        model = EmbeddingModel(model_name)

        for dataset in datasets:
            results = evaluate_on_dataset(
                dataset=dataset,
                model=model,
                metrics=metrics,
                verbose=verbose
            )

            all_results.append(results)

    # Create summary
    summary = create_benchmark_summary(all_results)

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_path = save_dir / 'benchmark_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for res in all_results:
                json_res = res.copy()
                if 'similarities' in json_res:
                    json_res['similarities'] = {
                        k: v.tolist() for k, v in json_res['similarities'].items()
                    }
                json_results.append(json_res)

            json.dump(json_results, f, indent=2)

        # Save summary
        summary_path = save_dir / 'benchmark_summary.csv'
        summary.to_csv(summary_path, index=False)

        if verbose:
            print(f"\nResults saved to {save_dir}")

    return {
        'results': all_results,
        'summary': summary
    }


def create_benchmark_summary(results: List[Dict]) -> pd.DataFrame:
    """
    Create summary DataFrame from benchmark results.

    Parameters
    ----------
    results : list of dict
        List of evaluation results

    Returns
    -------
    pd.DataFrame
        Summary dataframe
    """
    summary_rows = []

    for res in results:
        dataset_name = res['dataset_name']
        model_name = res['model_name']

        # Classification metrics
        if 'classification_metrics' in res:
            for metric, clf_metrics in res['classification_metrics'].items():
                row = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'similarity_metric': metric,
                }
                row.update(clf_metrics)
                summary_rows.append(row)

        # Correlation metrics
        if 'correlation_metrics' in res:
            for metric, corr_metrics in res['correlation_metrics'].items():
                # Find existing row or create new
                existing = [r for r in summary_rows
                           if r['dataset'] == dataset_name
                           and r['model'] == model_name
                           and r['similarity_metric'] == metric]

                if existing:
                    existing[0].update(corr_metrics)
                else:
                    row = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'similarity_metric': metric,
                    }
                    row.update(corr_metrics)
                    summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def compare_metrics_on_benchmark(
    results: List[Dict],
    primary_measure: str = 'accuracy'
) -> pd.DataFrame:
    """
    Compare different similarity metrics based on a primary performance measure.

    Parameters
    ----------
    results : list of dict
        Benchmark results
    primary_measure : str
        Primary measure to compare (e.g., 'accuracy', 'f1', 'pearson_r')

    Returns
    -------
    pd.DataFrame
        Comparison dataframe
    """
    comparison_data = []

    for res in results:
        if 'classification_metrics' in res:
            for metric, clf_metrics in res['classification_metrics'].items():
                if primary_measure in clf_metrics:
                    comparison_data.append({
                        'dataset': res['dataset_name'],
                        'model': res['model_name'],
                        'metric': metric,
                        primary_measure: clf_metrics[primary_measure]
                    })

    df = pd.DataFrame(comparison_data)

    # Pivot for easier comparison
    if len(df) > 0:
        pivot = df.pivot_table(
            values=primary_measure,
            index=['dataset', 'model'],
            columns='metric'
        )
        return pivot

    return df
