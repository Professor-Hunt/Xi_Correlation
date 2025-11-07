# Quick Start Guide

Get started with Chatterjee's Xi for semantic similarity in 5 minutes.

## Installation

```bash
git clone https://github.com/Professor-Hunt/Vector_Correlation.git
cd Vector_Correlation
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy, SciPy, pandas
- sentence-transformers
- PyTorch
- scikit-learn

All dependencies are listed in `requirements.txt`.

## Basic Usage

### 1. Compute Xi Similarity

```python
from src.similarity import symmetric_xi, EmbeddingModel

# Load embedding model
model = EmbeddingModel('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug."
]
embeddings = model.encode(sentences)

# Compute Xi similarity
similarity = symmetric_xi(embeddings[0], embeddings[1])
print(f"Xi similarity: {similarity:.3f}")
```

### 2. Compare with Cosine

```python
from src.similarity.metrics import cosine_similarity_score

cosine_sim = cosine_similarity_score(embeddings[0], embeddings[1])
print(f"Cosine similarity: {cosine_sim:.3f}")
print(f"Difference: {abs(similarity - cosine_sim):.3f}")
```

### 3. Batch Processing

```python
import numpy as np

sentences = [
    "First sentence",
    "Second sentence",
    "Third sentence"
]

embeddings = model.encode(sentences)

# Compute pairwise similarities
n = len(embeddings)
xi_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        xi_matrix[i, j] = symmetric_xi(embeddings[i], embeddings[j])

print(xi_matrix)
```

## Running Examples

### Basic Example

```bash
python examples/basic_usage.py
```

Shows Xi vs cosine on various sentence pairs.

### Benchmark Evaluation

```bash
python examples/benchmark_evaluation.py
```

Evaluates on STS-B dataset sample.

## Running Experiments

### Synthetic Experiments

```bash
python src/experiments/synthetic.py
```

Tests Xi on nonlinear transformations (N=17,500).

### All Experiments

```bash
python experiments/run_all_experiments.py
```

Runs complete experimental suite (~5 minutes).

## Understanding Results

### Xi Values

- **Range**: [0, 1] (can be slightly negative in finite samples)
- **1.0**: Perfect functional relationship
- **0.0**: Independence
- **Negative**: No relationship (finite-sample artifact)

### Comparison with Cosine

- **Xi typically < Cosine**: Xi is more conservative
- **Gap ~0.86%**: On benchmark data
- **Complementary**: Use together for robust similarity

## Next Steps

- **[API Reference](api_reference.md)**: Complete API documentation
- **[Experiments Guide](experiments.md)**: Reproduce all results
- **[Paper](../paper/semantic_similarity_paper.tex)**: Full technical details

## Common Issues

### Import Errors

If you get import errors, make sure you've installed in editable mode:

```bash
pip install -e .
```

### Model Download

First time running will download the sentence-transformers model (~90 MB). This is cached for future use.

### Memory Issues

For large datasets, process in batches:

```python
batch_size = 32
for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i+batch_size]
    embeddings = model.encode(batch)
    # Process batch...
```

## Getting Help

- **Issues**: https://github.com/Professor-Hunt/Vector_Correlation/issues
- **Discussions**: https://github.com/Professor-Hunt/Vector_Correlation/discussions
- **Paper**: See `paper/semantic_similarity_paper.tex` for detailed explanations
