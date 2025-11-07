#!/usr/bin/env python3
"""
Run additional experiments for paper:
1. STS-B benchmark evaluation
2. Hybrid cosine + Xi model
3. Runtime analysis

This script implements the "Future Work" section.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.experiments.stsb_data import load_stsb_sample
from src.experiments.benchmark import BenchmarkDataset, evaluate_on_dataset
from src.experiments.runtime_analysis import run_comprehensive_runtime_analysis
from src.similarity.embeddings import EmbeddingModel
from src.similarity.metrics import compute_all_similarities
from src.similarity.hybrid import evaluate_hybrid_model, optimize_hybrid_weights

# Setup plotting style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")


def experiment_1_stsb_benchmark(save_dir: Path):
    """
    Experiment 1: STS-B Benchmark Evaluation

    Evaluates cosine, xi, and xi_symmetric on STS-B data
    measuring correlation with human similarity judgments.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: STS-B BENCHMARK EVALUATION")
    print("=" * 70)

    # Load STS-B sample data
    stsb_data, description = load_stsb_sample()
    print(description)

    # Create benchmark dataset
    dataset = BenchmarkDataset(
        sentences1=stsb_data['sentence1'].tolist(),
        sentences2=stsb_data['sentence2'].tolist(),
        scores=stsb_data['score'].values,  # Continuous scores for correlation
        name='STS-B-Sample'
    )

    # Try TF-IDF first (faster, no downloads needed)
    print("\n" + "-" * 70)
    print("Evaluating with TF-IDF embeddings...")
    print("-" * 70)

    try:
        model_tfidf = EmbeddingModel('tfidf')
        results_tfidf = evaluate_on_dataset(
            dataset=dataset,
            model=model_tfidf,
            metrics=['cosine', 'xi', 'xi_symmetric'],
            verbose=True
        )

        # Save TF-IDF results
        if 'correlation_metrics' in results_tfidf:
            corr_df = pd.DataFrame(results_tfidf['correlation_metrics']).T
            corr_df.to_csv(save_dir / 'stsb_tfidf_correlations.csv')
            print("\nTF-IDF Correlations with Human Judgments:")
            print(corr_df[['spearman_r', 'pearson_r']].round(3))

    except Exception as e:
        print(f"TF-IDF evaluation failed: {e}")
        results_tfidf = None

    # Now try with BERT (may require download)
    print("\n" + "-" * 70)
    print("Attempting BERT evaluation (may take time for first run)...")
    print("-" * 70)

    results_bert = None
    try:
        model_bert = EmbeddingModel('all-MiniLM-L6-v2')
        results_bert = evaluate_on_dataset(
            dataset=dataset,
            model=model_bert,
            metrics=['cosine', 'xi', 'xi_symmetric'],
            verbose=True
        )

        # Save BERT results
        if 'correlation_metrics' in results_bert:
            corr_df = pd.DataFrame(results_bert['correlation_metrics']).T
            corr_df.to_csv(save_dir / 'stsb_bert_correlations.csv')
            print("\nBERT Correlations with Human Judgments:")
            print(corr_df[['spearman_r', 'pearson_r']].round(3))

            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            corr_df['spearman_r'].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title('Spearman Correlation with Human Similarity Judgments (STS-B)', fontsize=14, weight='bold')
            ax.set_xlabel('Similarity Metric', fontsize=12)
            ax.set_ylabel('Spearman Correlation (ρ)', fontsize=12)
            ax.set_ylim([0, 1.0])
            ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Strong correlation')
            ax.legend()
            plt.tight_layout()
            plt.savefig(save_dir / 'stsb_correlations.png', dpi=300, bbox_inches='tight')
            print(f"\nSaved correlation plot to {save_dir / 'stsb_correlations.png'}")

    except Exception as e:
        print(f"BERT evaluation skipped or failed: {e}")
        print("Note: This requires sentence-transformers library and model download.")

    return {
        'tfidf': results_tfidf,
        'bert': results_bert
    }


def experiment_2_hybrid_model(save_dir: Path):
    """
    Experiment 2: Hybrid Cosine + Xi Model

    Tests weighted combinations of cosine and Xi,
    finding optimal weights for different tasks.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: HYBRID COSINE + XI MODEL")
    print("=" * 70)

    # Load STS-B data for optimization
    stsb_data, _ = load_stsb_sample()

    # Create binary labels based on score threshold
    threshold = 3.0  # Moderate similarity
    binary_labels = (stsb_data['score'] >= threshold).astype(int)

    print(f"\nDataset: {len(stsb_data)} pairs")
    print(f"Binary labels (score >= {threshold}): {binary_labels.sum()} positive, {(1-binary_labels).sum()} negative")

    # Encode with TF-IDF first
    print("\n" + "-" * 70)
    print("Testing with TF-IDF embeddings...")
    print("-" * 70)

    try:
        model = EmbeddingModel('tfidf')

        # Encode all sentences
        all_sentences = stsb_data['sentence1'].tolist() + stsb_data['sentence2'].tolist()
        all_embeddings = model.encode(all_sentences)

        n = len(stsb_data)
        embeddings1 = all_embeddings[:n]
        embeddings2 = all_embeddings[n:]

        # Evaluate hybrid model
        print("\nEvaluating hybrid model with different weight combinations...")
        hybrid_results = evaluate_hybrid_model(
            embeddings1=embeddings1,
            embeddings2=embeddings2,
            labels=binary_labels,
            scores=stsb_data['score'].values,
            weights_to_test=np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
        )

        # Save results
        if 'classification' in hybrid_results:
            clf_df = pd.DataFrame(hybrid_results['classification'])
            clf_df.to_csv(save_dir / 'hybrid_classification_results.csv', index=False)

            print("\nClassification Performance vs Weight:")
            print(clf_df.round(3))

            # Find optimal
            best_idx = clf_df['accuracy'].idxmax()
            print(f"\nBest accuracy: {clf_df.loc[best_idx, 'accuracy']:.3f} at cosine_weight={clf_df.loc[best_idx, 'weights']:.1f}")

        if 'correlation' in hybrid_results:
            corr_df = pd.DataFrame(hybrid_results['correlation'])
            corr_df.to_csv(save_dir / 'hybrid_correlation_results.csv', index=False)

            print("\nCorrelation with Human Scores vs Weight:")
            print(corr_df.round(3))

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if 'classification' in hybrid_results:
            clf_df = pd.DataFrame(hybrid_results['classification'])
            ax1.plot(clf_df['weights'], clf_df['accuracy'], marker='o', linewidth=2, label='Accuracy')
            ax1.plot(clf_df['weights'], clf_df['f1'], marker='s', linewidth=2, label='F1 Score')
            ax1.set_xlabel('Cosine Weight (1 - weight = Xi weight)', fontsize=11)
            ax1.set_ylabel('Score', fontsize=11)
            ax1.set_title('Classification Performance vs Weight', fontsize=12, weight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, 1])

        if 'correlation' in hybrid_results:
            corr_df = pd.DataFrame(hybrid_results['correlation'])
            ax2.plot(corr_df['weights'], corr_df['spearman_r'], marker='o', linewidth=2, color='green')
            ax2.set_xlabel('Cosine Weight (1 - weight = Xi weight)', fontsize=11)
            ax2.set_ylabel('Spearman Correlation (ρ)', fontsize=11)
            ax2.set_title('Correlation with Human Judgments vs Weight', fontsize=12, weight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, 1])

        plt.tight_layout()
        plt.savefig(save_dir / 'hybrid_model_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved hybrid model plot to {save_dir / 'hybrid_model_analysis.png'}")

    except Exception as e:
        print(f"Hybrid model evaluation failed: {e}")
        hybrid_results = None

    return hybrid_results


def experiment_3_runtime_analysis(save_dir: Path):
    """
    Experiment 3: Runtime Performance Analysis

    Measures computational cost of different metrics
    across various dimensions and sample sizes.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: RUNTIME ANALYSIS")
    print("=" * 70)

    runtime_results = run_comprehensive_runtime_analysis(
        save_dir=save_dir,
        verbose=True
    )

    # Create visualizations
    print("\nCreating runtime visualizations...")

    # 1. Runtime vs Dimension
    if 'dimension_scaling' in runtime_results:
        df_dim = runtime_results['dimension_scaling']

        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in df_dim['metric'].unique():
            data = df_dim[df_dim['metric'] == metric]
            ax.plot(data['dimension'], data['mean_ms'], marker='o', linewidth=2, label=metric)

        ax.set_xlabel('Embedding Dimension', fontsize=12)
        ax.set_ylabel('Runtime (milliseconds)', fontsize=12)
        ax.set_title('Runtime vs Embedding Dimension', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(save_dir / 'runtime_vs_dimension.png', dpi=300, bbox_inches='tight')

    # 2. Comparison table visualization
    if 'comparison' in runtime_results:
        df_comp = runtime_results['comparison']

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_comp)))
        bars = ax.barh(df_comp['metric'], df_comp['mean_ms'], color=colors)
        ax.set_xlabel('Mean Runtime (milliseconds)', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        ax.set_title('Runtime Comparison (d=384, typical BERT dimension)', fontsize=14, weight='bold')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}ms',
                   ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_dir / 'runtime_comparison_bar.png', dpi=300, bbox_inches='tight')

    # 3. Pairwise scaling
    if 'pairwise_scaling' in runtime_results:
        df_pair = runtime_results['pairwise_scaling']

        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in df_pair['metric'].unique():
            data = df_pair[df_pair['metric'] == metric]
            ax.plot(data['n_pairs'], data['total_ms'], marker='o', linewidth=2, label=metric)

        ax.set_xlabel('Number of Pairs', fontsize=12)
        ax.set_ylabel('Total Runtime (milliseconds)', fontsize=12)
        ax.set_title('Pairwise Similarity Computation Scaling', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'runtime_pairwise_scaling.png', dpi=300, bbox_inches='tight')

    print(f"\nRuntime visualizations saved to {save_dir}")

    return runtime_results


def main():
    """Run all additional experiments."""
    print("\n" + "=" * 70)
    print("RUNNING ADDITIONAL EXPERIMENTS FOR PAPER")
    print("Implementing Future Work section")
    print("=" * 70)

    # Create output directories
    base_dir = Path('results/additional_experiments')
    exp1_dir = base_dir / 'stsb_benchmark'
    exp2_dir = base_dir / 'hybrid_model'
    exp3_dir = base_dir / 'runtime_analysis'

    for d in [exp1_dir, exp2_dir, exp3_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Run experiments
    results = {}

    # Experiment 1: STS-B Benchmark
    try:
        results['stsb'] = experiment_1_stsb_benchmark(exp1_dir)
    except Exception as e:
        print(f"\nExperiment 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 2: Hybrid Model
    try:
        results['hybrid'] = experiment_2_hybrid_model(exp2_dir)
    except Exception as e:
        print(f"\nExperiment 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 3: Runtime Analysis
    try:
        results['runtime'] = experiment_3_runtime_analysis(exp3_dir)
    except Exception as e:
        print(f"\nExperiment 3 failed: {e}")
        import traceback
        traceback.print_exc()

    # Create summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ADDITIONAL EXPERIMENTS")
    print("=" * 70)

    print(f"\nResults saved to: {base_dir}")
    print("\nGenerated files:")
    print(f"  STS-B: {list(exp1_dir.glob('*'))}")
    print(f"  Hybrid: {list(exp2_dir.glob('*'))}")
    print(f"  Runtime: {list(exp3_dir.glob('*'))}")

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
