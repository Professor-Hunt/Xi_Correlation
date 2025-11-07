"""
Experiment modules for evaluating similarity metrics.

Includes:
- Benchmark experiments (STS-B, SICK, etc.)
- Synthetic data experiments
- RAG simulation experiments
"""

from .benchmark import run_benchmark_evaluation
from .synthetic import run_synthetic_experiments
from .rag_simulation import run_rag_simulation

__all__ = [
    "run_benchmark_evaluation",
    "run_synthetic_experiments",
    "run_rag_simulation",
]
