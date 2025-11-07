"""
Retrieval-Augmented Generation (RAG) simulation experiments.

Simulates a document retrieval task to compare how different similarity
metrics rank relevant documents.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ..similarity.embeddings import EmbeddingModel
from ..similarity.metrics import rank_by_similarity


# Default knowledge base for testing
DEFAULT_KNOWLEDGE_BASE = [
    "The stock price increased significantly during the last quarter.",
    "She enjoys playing tennis on weekends.",
    "Rainfall has been heavy in the northern regions.",
    "The patient is not unhappy with the treatment.",
    "Wildflowers bloom beautifully in spring.",
    "Recent economic data shows strong consumer spending.",
    "Professional athletes train rigorously every day.",
    "Climate change is affecting precipitation patterns globally.",
    "Medical research continues to advance treatment options.",
    "Spring brings warmer temperatures and new growth.",
]

DEFAULT_QUERIES = [
    ("The patient is happy with the treatment.", 3),  # Paraphrase of doc 3
    ("Share prices rose a lot in the previous quarter.", 0),  # Paraphrase of doc 0
    ("Tennis is a fun sport to play.", 1),  # Related to doc 1
    ("It's raining a lot.", 2),  # Related to doc 2
]


class RAGSimulation:
    """
    RAG simulation for comparing retrieval performance.

    Parameters
    ----------
    knowledge_base : list of str
        Documents in the knowledge base
    queries : list of (str, int)
        List of (query, target_doc_index) tuples
    model_name : str
        Embedding model to use
    """

    def __init__(
        self,
        knowledge_base: List[str] = None,
        queries: List[Tuple[str, int]] = None,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        self.knowledge_base = knowledge_base or DEFAULT_KNOWLEDGE_BASE
        self.queries = queries or DEFAULT_QUERIES
        self.model = EmbeddingModel(model_name)

        # Precompute document embeddings
        self.doc_embeddings = self.model.encode(
            self.knowledge_base,
            show_progress_bar=False
        )

    def retrieve(
        self,
        query: str,
        metric: str = 'cosine',
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for a query.

        Parameters
        ----------
        query : str
            Query string
        metric : str
            Similarity metric to use
        top_k : int
            Number of documents to retrieve

        Returns
        -------
        list of (int, float)
            List of (doc_index, similarity_score) tuples
        """
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]

        rankings = rank_by_similarity(
            query=query_embedding,
            documents=self.doc_embeddings,
            metric=metric,
            top_k=top_k
        )

        return rankings

    def evaluate_retrieval(
        self,
        metrics: List[str] = ['cosine', 'xi', 'xi_symmetric'],
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate retrieval performance across multiple metrics.

        Parameters
        ----------
        metrics : list of str
            Metrics to evaluate
        top_k : int
            Number of documents to retrieve

        Returns
        -------
        dict
            Evaluation results including:
            - 'query_results': Per-query results
            - 'aggregate_metrics': Overall performance metrics
        """
        query_results = []

        for query_text, target_idx in self.queries:
            query_result = {
                'query': query_text,
                'target_doc': self.knowledge_base[target_idx],
                'target_idx': target_idx
            }

            for metric in metrics:
                rankings = self.retrieve(query_text, metric=metric, top_k=top_k)

                # Find position of target document
                positions = [i for i, (idx, score) in enumerate(rankings) if idx == target_idx]
                position = positions[0] if positions else -1

                # Get scores
                scores = [score for idx, score in rankings]
                retrieved_indices = [idx for idx, score in rankings]

                query_result[f'{metric}_position'] = position
                query_result[f'{metric}_in_top{top_k}'] = position >= 0
                query_result[f'{metric}_scores'] = scores
                query_result[f'{metric}_retrieved'] = retrieved_indices

            query_results.append(query_result)

        # Compute aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics(query_results, metrics, top_k)

        return {
            'query_results': query_results,
            'aggregate_metrics': aggregate_metrics
        }

    def _compute_aggregate_metrics(
        self,
        query_results: List[Dict],
        metrics: List[str],
        top_k: int
    ) -> Dict:
        """Compute aggregate performance metrics."""
        aggregate = {}

        for metric in metrics:
            positions = [qr[f'{metric}_position'] for qr in query_results]
            in_top_k = [qr[f'{metric}_in_top{top_k}'] for qr in query_results]

            # Mean Reciprocal Rank (MRR)
            rr_values = [1 / (pos + 1) if pos >= 0 else 0 for pos in positions]
            mrr = np.mean(rr_values)

            # Success@K (proportion of queries with target in top K)
            success_at_k = np.mean(in_top_k)

            # Mean rank (only for retrieved documents)
            retrieved_positions = [pos for pos in positions if pos >= 0]
            mean_rank = np.mean(retrieved_positions) if retrieved_positions else -1

            aggregate[metric] = {
                'mrr': float(mrr),
                f'success_at_{top_k}': float(success_at_k),
                'mean_rank': float(mean_rank),
                'n_queries': len(query_results)
            }

        return aggregate

    def visualize_retrieval(
        self,
        query_idx: int = 0,
        metrics: List[str] = ['cosine', 'xi'],
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Visualize retrieval results for a specific query.

        Parameters
        ----------
        query_idx : int
            Index of query to visualize
        metrics : list of str
            Metrics to compare
        top_k : int
            Number of documents to show

        Returns
        -------
        pd.DataFrame
            Table showing retrieved documents and their ranks
        """
        query_text, target_idx = self.queries[query_idx]

        print(f"Query: {query_text}")
        print(f"Target document (index {target_idx}): {self.knowledge_base[target_idx]}")
        print("\n")

        results = []
        for metric in metrics:
            rankings = self.retrieve(query_text, metric=metric, top_k=top_k)

            for rank, (doc_idx, score) in enumerate(rankings, 1):
                results.append({
                    'Metric': metric,
                    'Rank': rank,
                    'Doc Index': doc_idx,
                    'Score': f"{score:.4f}",
                    'Document': self.knowledge_base[doc_idx][:60] + "...",
                    'Is Target': '✓' if doc_idx == target_idx else ''
                })

        df = pd.DataFrame(results)
        return df


def run_rag_simulation(
    knowledge_base: Optional[List[str]] = None,
    queries: Optional[List[Tuple[str, int]]] = None,
    model_names: List[str] = ['all-MiniLM-L6-v2'],
    metrics: List[str] = ['cosine', 'xi', 'xi_symmetric'],
    top_k: int = 5,
    save_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict:
    """
    Run comprehensive RAG simulation experiment.

    Parameters
    ----------
    knowledge_base : list of str, optional
        Documents. If None, uses default knowledge base.
    queries : list of (str, int), optional
        Queries and target indices. If None, uses default queries.
    model_names : list of str
        Embedding models to test
    metrics : list of str
        Similarity metrics to evaluate
    top_k : int
        Number of documents to retrieve
    save_dir : Path, optional
        Directory to save results
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Comprehensive results for all models

    Examples
    --------
    >>> results = run_rag_simulation(
    ...     model_names=['all-MiniLM-L6-v2'],
    ...     metrics=['cosine', 'xi']
    ... )
    """
    all_results = []

    for model_name in model_names:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running RAG simulation with model: {model_name}")
            print('='*60)

        sim = RAGSimulation(
            knowledge_base=knowledge_base,
            queries=queries,
            model_name=model_name
        )

        results = sim.evaluate_retrieval(metrics=metrics, top_k=top_k)
        results['model_name'] = model_name

        all_results.append(results)

        if verbose:
            print("\nAggregate Metrics:")
            for metric, scores in results['aggregate_metrics'].items():
                print(f"\n{metric}:")
                for score_name, score_value in scores.items():
                    print(f"  {score_name}: {score_value:.4f}")

    # Create summary
    summary = create_rag_summary(all_results)

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = save_dir / 'rag_summary.csv'
        summary.to_csv(summary_path, index=False)

        # Save detailed results
        import json
        results_path = save_dir / 'rag_detailed_results.json'

        # Prepare for JSON serialization
        json_results = []
        for res in all_results:
            json_res = res.copy()
            # Remove non-serializable parts
            if 'query_results' in json_res:
                for qr in json_res['query_results']:
                    for key in list(qr.keys()):
                        if 'scores' in key or 'retrieved' in key:
                            if isinstance(qr[key], list):
                                qr[key] = [float(x) if isinstance(x, (np.floating, float)) else int(x)
                                          for x in qr[key]]
            json_results.append(json_res)

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        if verbose:
            print(f"\nResults saved to {save_dir}")

    return {
        'results': all_results,
        'summary': summary
    }


def create_rag_summary(results: List[Dict]) -> pd.DataFrame:
    """
    Create summary DataFrame from RAG simulation results.

    Parameters
    ----------
    results : list of dict
        List of RAG simulation results

    Returns
    -------
    pd.DataFrame
        Summary dataframe
    """
    summary_rows = []

    for res in results:
        model_name = res['model_name']
        aggregate = res['aggregate_metrics']

        for metric, scores in aggregate.items():
            row = {
                'model': model_name,
                'similarity_metric': metric,
            }
            row.update(scores)
            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def compare_rag_metrics(results: List[Dict], measure: str = 'mrr') -> pd.DataFrame:
    """
    Compare RAG performance across metrics.

    Parameters
    ----------
    results : list of dict
        RAG results
    measure : str
        Performance measure to compare (e.g., 'mrr', 'success_at_5')

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    summary = create_rag_summary(results)

    if measure in summary.columns:
        pivot = summary.pivot_table(
            values=measure,
            index='model',
            columns='similarity_metric'
        )
        return pivot

    return summary
