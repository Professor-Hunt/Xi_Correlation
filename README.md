# Chatterjee's Xi for Semantic Similarity

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A validated rank-based alternative to cosine similarity for measuring semantic similarity in sentence embeddings.**

This repository provides a complete implementation of applying Chatterjee's rank correlation coefficient (ξ) to BERT sentence embeddings, achieving **0.859 Spearman correlation** with human judgments on 1,500 STS-B benchmark pairs—within **0.86%** of cosine similarity.

## 📊 Key Results

- **ρ = 0.859** Spearman correlation on STS-B validation (1,500 pairs)
- **Within 0.86%** of cosine similarity performance
- **82.8% accuracy** on binary similarity classification
- **Validated mechanism**: Distributed signal aggregation across 25% of embedding dimensions
- **Theoretical foundation**: Near-perfect detection of nonlinear relationships (ξ ≥ 0.93) on synthetic data

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Professor-Hunt/Xi_Correlation.git
cd Xi_Correlation
pip install -e .
```

### Basic Usage

```python
from src.similarity import chatterjee_xi, symmetric_xi, EmbeddingModel
from src.similarity.metrics import cosine_similarity_score

# Load a sentence embedding model
model = EmbeddingModel('all-MiniLM-L6-v2')

# Encode two sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A swift auburn fox leaps over a sleepy canine."
]
embeddings = model.encode(sentences)

# Compute similarities
xi = symmetric_xi(embeddings[0], embeddings[1])
cosine = cosine_similarity_score(embeddings[0], embeddings[1])

print(f"Xi similarity: {xi:.3f}")
print(f"Cosine similarity: {cosine:.3f}")
```

**Output:**
```
Xi similarity: 0.279
Cosine similarity: 0.704
```

## 📖 What is Chatterjee's Xi?

Chatterjee's ξ is a **rank-based correlation coefficient** that detects both linear and nonlinear functional relationships between variables. Unlike Pearson or cosine similarity, which measure linear/magnitude relationships, ξ captures whether one variable is a (possibly nonlinear) function of another.

### Key Properties

- **Range**: [0, 1] (population), can be slightly negative in finite samples
- **Equals 1**: When one variable is a function of the other
- **Equals 0**: When variables are independent
- **Detects**: Both monotonic and non-monotonic relationships
- **Complexity**: O(n log n) for computation

## 🎯 Why Use Xi for Semantic Similarity?

### Advantages

- ✅ **Validated performance**: Within 1% of cosine on benchmark data
- ✅ **Rank-based**: More robust to magnitude variations
- ✅ **Conservative**: Stricter similarity requirements (useful for high-precision retrieval)
- ✅ **Complementary**: Provides different perspective than cosine
- ✅ **Theoretically grounded**: Detects nonlinear relationships

### When to Use Xi vs Cosine

**Use ξ when:**
- Conservative similarity judgments are needed
- Low-similarity discrimination is important
- Rank structure matters more than magnitude
- Building ensemble methods

**Use cosine when:**
- Maximum performance is critical (0.86% advantage)
- High-similarity discrimination is needed
- Computational speed is paramount
- Magnitude information is valuable

## 📂 Repository Structure

```
Xi_Correlation/
├── src/                    # Core implementation
│   ├── similarity/         # Similarity metrics
│   ├── experiments/        # Experiment modules
│   └── utils/             # Utilities
├── experiments/           # Reproducible experiments
├── tests/                 # Unit tests
├── paper/                 # LaTeX paper
├── notebooks/             # Jupyter examples
├── examples/              # Quick start examples
├── results/               # Experimental results
└── docs/                  # Documentation
```

## 🔬 Experiments

### Reproduce Main Results

```bash
# Run all experiments (synthetic + benchmarks)
python experiments/run_all_experiments.py

# Run additional experiments (STS-B, hybrid models, runtime)
python experiments/run_additional_experiments.py
```

**Expected runtime:** ~5 minutes total

### Results Summary

| Dataset | Metric | Xi | Cosine | Gap |
|---------|--------|-----|--------|-----|
| STS-B (1,500 pairs) | Spearman ρ | 0.8586 | 0.8672 | 0.86% |
| STS-B (1,500 pairs) | Pearson r | 0.8337 | 0.8696 | 3.59% |
| STS-B (1,500 pairs) | Accuracy | 82.8% | 83.6% | 0.8% |
| Synthetic (17,500) | Quadratic | 0.988 | 0.061 | +92.7pp |
| Synthetic (17,500) | Absolute | 0.988 | 0.036 | +95.2pp |

## 📚 Documentation

- **[Quick Start Guide](docs/quickstart.md)**: Get started in 5 minutes
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Experiments Guide](docs/experiments.md)**: Reproduce all results
- **[Paper](paper/semantic_similarity_paper.tex)**: Full technical details

## 🔑 Core API

### Similarity Metrics

```python
from src.similarity import chatterjee_xi, symmetric_xi
from src.similarity.metrics import cosine_similarity_score

# Asymmetric Xi (measures if Y is a function of X)
xi_xy = chatterjee_xi(x, y)

# Symmetric Xi (max of both directions)
s_xi = symmetric_xi(x, y)

# Cosine similarity
cos_sim = cosine_similarity_score(x, y)
```

### Embedding Models

```python
from src.similarity import EmbeddingModel

# Initialize model
model = EmbeddingModel('all-MiniLM-L6-v2')

# Encode sentences
embeddings = model.encode([
    "First sentence",
    "Second sentence"
])

# Supported models:
# - 'all-MiniLM-L6-v2' (fast, 384-dim)
# - 'all-mpnet-base-v2' (quality, 768-dim)
# - 'tfidf' (baseline)
# - 'lsa' (baseline)
```

## 📊 Mechanistic Understanding

Our research reveals **why** dimensionwise ξ works:

1. **Distributed aggregation**: ~95 dimensions (25%) contribute meaningfully
2. **No dominance**: Strongest dimension shows only 0.228 correlation
3. **Conservative behavior**: ξ < cosine in 99.3% of pairs
4. **Rank structure**: Captures 99% of semantic signal through rank alone

See [paper](paper/semantic_similarity_paper.tex) for full mechanistic analysis.

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_chatterjee_xi.py
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{hunt2025chatterjee,
  title={Beyond Cosine: A Rank-Based Measure of Semantic Similarity Using Chatterjee's Xi},
  author={Hunt, Joshua O.S. and Hunt, Emily J. and Neupane, Prashant},
  journal={arXiv preprint},
  year={2025}
}
```

And the original Chatterjee paper:

```bibtex
@article{chatterjee2021new,
  title={A New Coefficient of Correlation},
  author={Chatterjee, Sourav},
  journal={Journal of the American Statistical Association},
  volume={116},
  number={536},
  pages={2009--2022},
  year={2021}
}
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Sourav Chatterjee for the original ξ correlation coefficient
- The sentence-transformers team for excellent embedding models
- The HuggingFace team for datasets and model hosting

## 📞 Contact

**Joshua O.S. Hunt, Emily J. Hunt, Prashant Neupane**
- GitHub: [@Professor-Hunt](https://github.com/Professor-Hunt)
- Repository: [Xi_Correlation](https://github.com/Professor-Hunt/Xi_Correlation)

---

**Note**: This is research software. While extensively tested, use at your own discretion for production applications.
