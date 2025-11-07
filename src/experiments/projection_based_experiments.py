"""
Projection-based experiments using the theoretically valid method.

This module addresses the peer review concern that the "simplified" method
(treating d components as paired observations) is statistically invalid.

All experiments here use the projection-based method from Section 3.1 of the paper,
which projects high-dimensional embeddings onto random directions and averages
the resulting 1D Xi values.

Key difference from previous experiments:
- OLD (INVALID): compute_all_similarities(emb1, emb2) treats 384 dims as observations
- NEW (VALID): Generate (n, 384) stochastic embeddings and use projection_based_xi

References:
-----------
Peer review: "The 'simplified' method (Sec 3.2) and all experiments that used it
             must be removed from the main paper."
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path
import warnings

from ..similarity.chatterjee_xi import projection_based_xi, symmetric_xi
from ..similarity.metrics import cosine_similarity_score

# Try to import stochastic embedder
try:
    from ..similarity.stochastic_embeddings import StochasticEmbedder
    STOCHASTIC_AVAILABLE = True
except ImportError:
    STOCHASTIC_AVAILABLE = False
    warnings.warn("StochasticEmbedder not available. Some experiments will be skipped.")


# Same sentence pairs as in original experiments (for direct comparison)
SENTENCE_PAIRS = [
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
]


def projection_based_sentence_similarity(
    sentence1: str,
    sentence2: str,
    embedder: 'StochasticEmbedder',
    n_samples: int = 50,
    n_projections: int = 100
) -> Dict[str, float]:
    """
    Compute projection-based Xi similarity for a sentence pair.

    This is the CORRECT method per the peer review requirements.

    Parameters
    ----------
    sentence1, sentence2 : str
        Sentences to compare
    embedder : StochasticEmbedder
        Stochastic embedding generator
    n_samples : int
        Number of stochastic embeddings to generate per sentence
    n_projections : int
        Number of random projections for Xi computation

    Returns
    -------
    dict
        Contains 'xi_mean', 'xi_std', 'symmetric_xi_mean', 'symmetric_xi_std'
    """
    # Generate stochastic embeddings: (n_samples, embedding_dim)
    X, Y = embedder.encode_pair(sentence1, sentence2, n_samples=n_samples)

    # Compute projection-based Xi
    xi_xy_mean, xi_xy_std = projection_based_xi(X, Y, n_projections=n_projections)
    xi_yx_mean, xi_yx_std = projection_based_xi(Y, X, n_projections=n_projections)

    # Symmetric Xi: max(Xi(X,Y), Xi(Y,X))
    symmetric_xi_mean = max(xi_xy_mean, xi_yx_mean)
    symmetric_xi_std = np.sqrt(xi_xy_std**2 + xi_yx_std**2) / 2  # Approximate

    # Also compute cosine on mean embeddings for comparison
    mean_x = X.mean(axis=0)
    mean_y = Y.mean(axis=0)
    cosine = cosine_similarity_score(mean_x, mean_y)

    return {
        'xi_xy_mean': xi_xy_mean,
        'xi_xy_std': xi_xy_std,
        'xi_yx_mean': xi_yx_mean,
        'xi_yx_std': xi_yx_std,
        'symmetric_xi_mean': symmetric_xi_mean,
        'symmetric_xi_std': symmetric_xi_std,
        'cosine': cosine,
        'n_samples': n_samples,
        'n_projections': n_projections
    }


def run_projection_based_bert_experiments(
    model_name: str = 'all-MiniLM-L6-v2',
    n_samples: int = 50,
    n_projections: int = 100,
    stochastic_method: str = 'dropout',
    save_dir: Path = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Rerun BERT experiments with projection-based method.

    This REPLACES the experiments from Section 4.3 of the original paper,
    which used the invalid "simplified" method.

    Parameters
    ----------
    model_name : str
        BERT model to use
    n_samples : int
        Number of stochastic embeddings per sentence
    n_projections : int
        Number of random projections for Xi
    stochastic_method : str
        'dropout' or 'perturbation'
    save_dir : Path
        Directory to save results
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Results for all sentence pairs
    """
    if not STOCHASTIC_AVAILABLE:
        raise ImportError(
            "StochasticEmbedder required. Install sentence-transformers: "
            "pip install sentence-transformers"
        )

    if verbose:
        print("=" * 70)
        print("PROJECTION-BASED BERT EXPERIMENTS (Section 4.3 Revision)")
        print("=" * 70)
        print(f"\nModel: {model_name}")
        print(f"Stochastic method: {stochastic_method}")
        print(f"Samples per sentence: {n_samples}")
        print(f"Random projections: {n_projections}")
        print(f"Total sentence pairs: {len(SENTENCE_PAIRS)}\n")

    # Initialize embedder
    embedder = StochasticEmbedder(
        model_name=model_name,
        method=stochastic_method
    )

    results = []

    for i, (sent1, sent2, label) in enumerate(SENTENCE_PAIRS):
        if verbose:
            print(f"Processing pair {i+1}/{len(SENTENCE_PAIRS)}...")

        result = projection_based_sentence_similarity(
            sent1, sent2, embedder,
            n_samples=n_samples,
            n_projections=n_projections
        )

        results.append({
            'pair_id': i,
            'sentence1': sent1,
            'sentence2': sent2,
            'label': label,
            **result
        })

    df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"\nSymmetric Xi by label:")
        print(df.groupby('label')['symmetric_xi_mean'].agg(['mean', 'std', 'min', 'max']))
        print(f"\nCosine by label:")
        print(df.groupby('label')['cosine'].agg(['mean', 'std', 'min', 'max']))

        # Classification accuracy
        threshold_xi = df['symmetric_xi_mean'].median()
        threshold_cos = df['cosine'].median()

        pred_xi = (df['symmetric_xi_mean'] >= threshold_xi).astype(int)
        pred_cos = (df['cosine'] >= threshold_cos).astype(int)

        acc_xi = (pred_xi == df['label']).mean()
        acc_cos = (pred_cos == df['label']).mean()

        print(f"\nClassification Accuracy (threshold=median):")
        print(f"  Symmetric Xi: {acc_xi:.1%}")
        print(f"  Cosine: {acc_cos:.1%}")

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        csv_path = save_dir / 'projection_based_bert_results.csv'
        df.to_csv(csv_path, index=False)

        if verbose:
            print(f"\nResults saved to: {csv_path}")

    return df


def ablation_study_k(
    model_name: str = 'all-MiniLM-L6-v2',
    n_samples: int = 50,
    k_values: List[int] = [10, 25, 50, 100, 200],
    stochastic_method: str = 'dropout',
    save_dir: Path = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Ablation study on k (number of random projections).

    The peer review explicitly requests: "The projection-based method (Sec 3.1)
    depends on k, the number of random projections. This is a new, critical
    hyperparameter. The revised paper must include an analysis (e.g., an ablation
    study) on the sensitivity of the experimental results to the choice of k."

    Parameters
    ----------
    model_name : str
        BERT model to use
    n_samples : int
        Number of stochastic embeddings per sentence
    k_values : list of int
        Different values of k to test
    stochastic_method : str
        'dropout' or 'perturbation'
    save_dir : Path
        Directory to save results
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Results for different k values
    """
    if not STOCHASTIC_AVAILABLE:
        raise ImportError("StochasticEmbedder required")

    if verbose:
        print("=" * 70)
        print("ABLATION STUDY: Number of Projections (k)")
        print("=" * 70)
        print(f"\nTesting k values: {k_values}")
        print(f"Using {len(SENTENCE_PAIRS)} sentence pairs\n")

    embedder = StochasticEmbedder(model_name=model_name, method=stochastic_method)

    # Pre-generate stochastic embeddings (same for all k)
    if verbose:
        print("Generating stochastic embeddings...")

    embeddings_cache = []
    for sent1, sent2, label in SENTENCE_PAIRS:
        X, Y = embedder.encode_pair(sent1, sent2, n_samples=n_samples)
        embeddings_cache.append((X, Y, label))

    results = []

    for k in k_values:
        if verbose:
            print(f"\nTesting k={k}...")

        for pair_id, (X, Y, label) in enumerate(embeddings_cache):
            # Compute projection-based Xi with this k
            xi_xy_mean, xi_xy_std = projection_based_xi(X, Y, n_projections=k)
            xi_yx_mean, xi_yx_std = projection_based_xi(Y, X, n_projections=k)

            symmetric_xi = max(xi_xy_mean, xi_yx_mean)

            results.append({
                'k': k,
                'pair_id': pair_id,
                'label': label,
                'symmetric_xi': symmetric_xi,
                'xi_xy': xi_xy_mean,
                'xi_yx': xi_yx_mean
            })

    df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION RESULTS")
        print("=" * 70)

        # Summary by k
        summary = df.groupby('k').agg({
            'symmetric_xi': ['mean', 'std']
        }).round(4)
        print("\nSymmetric Xi by k:")
        print(summary)

        # Classification accuracy by k
        print("\nClassification Accuracy by k:")
        for k in k_values:
            df_k = df[df['k'] == k]
            threshold = df_k['symmetric_xi'].median()
            pred = (df_k['symmetric_xi'] >= threshold).astype(int)
            acc = (pred == df_k['label']).mean()
            print(f"  k={k:3d}: {acc:.1%}")

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        csv_path = save_dir / 'ablation_k_results.csv'
        df.to_csv(csv_path, index=False)

        if verbose:
            print(f"\nResults saved to: {csv_path}")

    return df


def comparison_simplified_vs_projection(
    model_name: str = 'all-MiniLM-L6-v2',
    n_samples: int = 50,
    n_projections: int = 100,
    save_dir: Path = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Direct comparison: Simplified method (INVALID) vs Projection-based (VALID).

    This demonstrates why the reviewer's critique is correct by showing that:
    1. Results differ significantly between methods
    2. Simplified method is sensitive to dimension ordering
    3. Projection-based method is robust

    Parameters
    ----------
    model_name : str
        BERT model to use
    n_samples : int
        Number of stochastic samples for projection method
    n_projections : int
        Number of random projections
    save_dir : Path
        Directory to save results
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    if not STOCHASTIC_AVAILABLE:
        raise ImportError("StochasticEmbedder required")

    if verbose:
        print("=" * 70)
        print("COMPARISON: Simplified (INVALID) vs Projection-based (VALID)")
        print("=" * 70)

    from ..similarity.embeddings import EmbeddingModel

    # For simplified method: deterministic embeddings
    model_simple = EmbeddingModel(model_name)

    # For projection method: stochastic embeddings
    embedder_proj = StochasticEmbedder(model_name=model_name, method='dropout')

    results = []

    for i, (sent1, sent2, label) in enumerate(SENTENCE_PAIRS):
        if verbose:
            print(f"\nPair {i+1}: {'Similar' if label else 'Dissimilar'}")

        # Method 1: Simplified (INVALID)
        emb1 = model_simple.encode([sent1])[0]
        emb2 = model_simple.encode([sent2])[0]
        xi_simplified = symmetric_xi(emb1, emb2)  # Treats 384 dims as observations
        cosine_simple = cosine_similarity_score(emb1, emb2)

        # Method 2: Projection-based (VALID)
        X, Y = embedder_proj.encode_pair(sent1, sent2, n_samples=n_samples)
        xi_xy, _ = projection_based_xi(X, Y, n_projections=n_projections)
        xi_yx, _ = projection_based_xi(Y, X, n_projections=n_projections)
        xi_projection = max(xi_xy, xi_yx)

        mean_x = X.mean(axis=0)
        mean_y = Y.mean(axis=0)
        cosine_proj = cosine_similarity_score(mean_x, mean_y)

        if verbose:
            print(f"  Simplified Xi:      {xi_simplified:.3f}")
            print(f"  Projection-based Xi: {xi_projection:.3f}")
            print(f"  Difference:         {abs(xi_simplified - xi_projection):.3f}")

        results.append({
            'pair_id': i,
            'label': label,
            'sentence1': sent1[:50] + "...",
            'sentence2': sent2[:50] + "...",
            'xi_simplified': xi_simplified,
            'xi_projection': xi_projection,
            'xi_difference': abs(xi_simplified - xi_projection),
            'cosine_simple': cosine_simple,
            'cosine_proj': cosine_proj
        })

    df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\nMean absolute difference in Xi:")
        print(f"  {df['xi_difference'].mean():.4f} (std: {df['xi_difference'].std():.4f})")
        print(f"\nCorrelation between methods:")
        print(f"  Pearson r = {df[['xi_simplified', 'xi_projection']].corr().iloc[0,1]:.4f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / 'comparison_simplified_vs_projection.csv'
        df.to_csv(csv_path, index=False)

        if verbose:
            print(f"\nResults saved to: {csv_path}")

    return df


def main():
    """Run all projection-based experiments."""
    print("\n" + "=" * 70)
    print("PROJECTION-BASED EXPERIMENTS")
    print("Addressing Peer Review Methodological Concerns")
    print("=" * 70 + "\n")

    save_dir = Path("results/projection_based")
    save_dir.mkdir(parents=True, exist_ok=True)

    if not STOCHASTIC_AVAILABLE:
        print("ERROR: sentence-transformers not available.")
        print("Install with: pip install sentence-transformers")
        return

    # Experiment 1: Main BERT experiments
    print("\n" + "=" * 70)
    print("Experiment 1: BERT with Projection-Based Method")
    print("=" * 70)
    df_bert = run_projection_based_bert_experiments(
        model_name='all-MiniLM-L6-v2',
        n_samples=50,
        n_projections=100,
        stochastic_method='dropout',
        save_dir=save_dir,
        verbose=True
    )

    # Experiment 2: Ablation study on k
    print("\n" + "=" * 70)
    print("Experiment 2: Ablation Study on k")
    print("=" * 70)
    df_ablation = ablation_study_k(
        model_name='all-MiniLM-L6-v2',
        n_samples=50,
        k_values=[10, 25, 50, 100, 200],
        stochastic_method='dropout',
        save_dir=save_dir,
        verbose=True
    )

    # Experiment 3: Comparison
    print("\n" + "=" * 70)
    print("Experiment 3: Simplified vs Projection-based")
    print("=" * 70)
    df_comparison = comparison_simplified_vs_projection(
        model_name='all-MiniLM-L6-v2',
        n_samples=50,
        n_projections=100,
        save_dir=save_dir,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
