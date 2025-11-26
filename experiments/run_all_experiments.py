#!/usr/bin/env python3
"""
Main script to run all experiments for the research project.

This script runs:
1. Synthetic experiments
2. Benchmark evaluations
3. RAG simulations
4. Generates figures for the paper

Results are saved to the results/ and figures/ directories.
"""

import sys
from pathlib import Path
import argparse
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.synthetic import run_synthetic_experiments, visualize_synthetic_relationships
from src.experiments.benchmark import run_benchmark_evaluation, BenchmarkDataset
from src.experiments.rag_simulation import run_rag_simulation
from src.utils.visualization import (
    plot_metric_comparison,
    plot_performance_comparison,
    plot_synthetic_experiments
)
import pandas as pd


def setup_directories():
    """Create necessary directories."""
    (project_root / 'results').mkdir(exist_ok=True)
    (project_root / 'figures').mkdir(exist_ok=True)
    print("✓ Directories created")


def run_synthetic(args):
    """Run synthetic experiments."""
    print("\n" + "="*80)
    print("RUNNING SYNTHETIC EXPERIMENTS")
    print("="*80)

    results = run_synthetic_experiments(
        n_samples=args.n_samples,
        n_repetitions=args.n_repetitions,
        random_state=args.seed,
        save_dir=project_root / 'results' / 'synthetic'
    )

    # Create visualization
    print("\nGenerating synthetic experiments figure...")
    fig = visualize_synthetic_relationships(
        n_samples=200,
        random_state=args.seed,
        save_path=project_root / 'figures' / 'correlation_comparison.png'
    )

    # Create summary plot
    from src.utils.visualization import plot_synthetic_experiments
    fig = plot_synthetic_experiments(
        results,
        save_path=project_root / 'figures' / 'synthetic_experiments.png'
    )

    print("✓ Synthetic experiments complete")
    return results


def run_benchmark(args):
    """Run benchmark evaluations."""
    print("\n" + "="*80)
    print("RUNNING BENCHMARK EVALUATIONS")
    print("="*80)

    # Load default dataset
    dataset = BenchmarkDataset.load_default()

    results = run_benchmark_evaluation(
        datasets=[dataset],
        model_names=args.models,
        metrics=['cosine', 'xi', 'xi_symmetric', 'pearson', 'spearman'],
        save_dir=project_root / 'results' / 'benchmark',
        verbose=True
    )

    # Create visualizations
    print("\nGenerating benchmark comparison figures...")

    # Get a results dataframe for visualization
    if len(results['results']) > 0:
        res = results['results'][0]
        similarities = res['similarities']

        # Create comparison dataframe
        pairs_data = []
        for i in range(len(similarities['cosine'])):
            pairs_data.append({
                'pair_idx': i,
                'cosine': similarities['cosine'][i],
                'xi': similarities['xi'][i],
                'xi_symmetric': similarities.get('xi_symmetric', [0]*len(similarities['cosine']))[i]
            })
        df = pd.DataFrame(pairs_data)

        # Cosine vs Xi plot
        fig = plot_metric_comparison(
            df,
            x_metric='cosine',
            y_metric='xi',
            title='Cosine vs Xi Similarity',
            save_path=project_root / 'figures' / 'cosine_vs_xi.png'
        )

    print("✓ Benchmark evaluations complete")
    return results


def run_rag(args):
    """Run RAG simulations."""
    print("\n" + "="*80)
    print("RUNNING RAG SIMULATIONS")
    print("="*80)

    results = run_rag_simulation(
        model_names=args.models,
        metrics=['cosine', 'xi', 'xi_symmetric'],
        top_k=5,
        save_dir=project_root / 'results' / 'rag',
        verbose=True
    )

    # Create performance comparison
    print("\nGenerating RAG performance figure...")

    summary = results['summary']
    if len(summary) > 0:
        # Create performance comparison dict
        perf_dict = {}
        for _, row in summary.iterrows():
            metric = row['similarity_metric']
            if metric not in perf_dict:
                perf_dict[metric] = {}
            perf_dict[metric]['MRR'] = row['mrr']
            if 'success_at_5' in row:
                perf_dict[metric]['Success@5'] = row['success_at_5']

        if perf_dict:
            fig = plot_performance_comparison(
                perf_dict,
                title='RAG Retrieval Performance Comparison',
                save_path=project_root / 'figures' / 'rag_performance.png'
            )

    print("✓ RAG simulations complete")
    return results


def generate_paper_tables(synthetic_results, benchmark_results, rag_results):
    """Generate LaTeX tables for the paper."""
    print("\n" + "="*80)
    print("GENERATING PAPER TABLES")
    print("="*80)

    tables_dir = project_root / 'paper' / 'tables'
    tables_dir.mkdir(exist_ok=True, parents=True)

    # Synthetic results table
    if 'summary' in synthetic_results:
        summary = synthetic_results['summary']
        latex_table = summary.to_latex(
            index=False,
            float_format='%.3f',
            columns=['relationship', 'cosine_mean', 'xi_mean', 'pearson_mean', 'spearman_mean'],
            header=['Relationship', 'Cosine', '$\\xi$', 'Pearson', 'Spearman'],
            escape=False
        )

        with open(tables_dir / 'synthetic_summary.tex', 'w') as f:
            f.write(latex_table)

        print(f"✓ Saved synthetic summary table")

    # Benchmark results table
    if 'summary' in benchmark_results:
        summary = benchmark_results['summary']
        if 'accuracy' in summary.columns:
            latex_table = summary.to_latex(
                index=False,
                float_format='%.3f',
                escape=False
            )

            with open(tables_dir / 'benchmark_summary.tex', 'w') as f:
                f.write(latex_table)

            print(f"✓ Saved benchmark summary table")

    # RAG results table
    if 'summary' in rag_results:
        summary = rag_results['summary']
        latex_table = summary.to_latex(
            index=False,
            float_format='%.3f',
            escape=False
        )

        with open(tables_dir / 'rag_summary.tex', 'w') as f:
            f.write(latex_table)

        print(f"✓ Saved RAG summary table")


def main():
    parser = argparse.ArgumentParser(
        description='Run all experiments for Xi Correlation research project'
    )

    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['synthetic', 'benchmark', 'rag', 'all'],
        default=['all'],
        help='Which experiments to run'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['all-MiniLM-L6-v2'],
        help='Embedding models to evaluate'
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=500,
        help='Number of samples for synthetic experiments'
    )

    parser.add_argument(
        '--n-repetitions',
        type=int,
        default=10,
        help='Number of repetitions for synthetic experiments'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--skip-figures',
        action='store_true',
        help='Skip figure generation'
    )

    args = parser.parse_args()

    print("Xi Correlation Research Project")
    print("Experiment Runner")
    print("="*80)

    start_time = time.time()

    # Setup
    setup_directories()

    # Determine which experiments to run
    experiments = args.experiments
    if 'all' in experiments:
        experiments = ['synthetic', 'benchmark', 'rag']

    # Run experiments
    synthetic_results = None
    benchmark_results = None
    rag_results = None

    if 'synthetic' in experiments:
        synthetic_results = run_synthetic(args)

    if 'benchmark' in experiments:
        benchmark_results = run_benchmark(args)

    if 'rag' in experiments:
        rag_results = run_rag(args)

    # Generate paper tables
    if not args.skip_figures:
        if synthetic_results or benchmark_results or rag_results:
            generate_paper_tables(
                synthetic_results or {},
                benchmark_results or {},
                rag_results or {}
            )

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nResults saved to: {project_root / 'results'}")
    print(f"Figures saved to: {project_root / 'figures'}")
    print(f"Paper tables saved to: {project_root / 'paper' / 'tables'}")


if __name__ == '__main__':
    main()
