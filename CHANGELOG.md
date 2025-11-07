# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-07

### Added
- Initial public release
- Complete implementation of Chatterjee's Xi for semantic similarity
- Dimensionwise Xi method (validated on 1,500 STS-B pairs)
- Projection-based Xi method (validated on synthetic data)
- Comprehensive experimental validation (N=17,500+ observations)
- STS-B benchmark evaluation (ρ=0.859)
- Mechanistic analysis of dimensionwise Xi
- Synthetic nonlinear relationship experiments
- Hybrid model implementation (cosine + Xi)
- Runtime analysis and optimization
- Complete test suite with >90% coverage
- Example scripts and documentation
- LaTeX paper with full technical details
- API reference documentation

### Performance
- Spearman ρ = 0.8586 on STS-B validation (1,500 pairs)
- Within 0.86% of cosine similarity performance
- Binary classification accuracy: 82.8%
- O(d log d) computational complexity

### Documentation
- Comprehensive README with quick start guide
- API reference documentation
- Experiment reproduction guide
- Example scripts for common use cases
- Full LaTeX paper with theoretical foundations

## [Unreleased]

### Planned
- Additional benchmark evaluations (SICK-R, MS MARCO)
- Support for other embedding models (RoBERTa, DeBERTa)
- Optimized batch processing
- GPU acceleration for large-scale similarity computation
- Interactive visualization tools
- Web API for similarity computation

---

For detailed changes, see the [commit history](https://github.com/Professor-Hunt/Vector_Correlation/commits/main).
