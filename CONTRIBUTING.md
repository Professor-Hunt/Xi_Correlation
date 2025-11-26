# Contributing to Xi Correlation

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Xi_Correlation.git
   cd Xi_Correlation
   ```
3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use prefixes:
- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation
- `test/` for test additions

### 2. Make Changes

- Write clear, documented code
- Follow PEP 8 style guidelines
- Add docstrings to all functions/classes
- Include type hints where appropriate

### 3. Add Tests

All new code should include tests:

```bash
pytest tests/ -v
```

Aim for >80% coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

### 4. Run Code Quality Checks

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### 5. Commit Changes

Write clear commit messages:

```bash
git commit -m "feat: Add support for batch similarity computation"
```

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Code

- Follow PEP 8
- Use Black for formatting (line length: 88)
- Use meaningful variable names
- Add docstrings (NumPy format)

Example:

```python
def symmetric_xi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute symmetric Chatterjee's Xi correlation.

    Parameters
    ----------
    x : np.ndarray
        First vector (1D array)
    y : np.ndarray
        Second vector (1D array)

    Returns
    -------
    float
        Symmetric Xi value in [0, 1]

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> symmetric_xi(x, y)
    0.946
    """
    return max(chatterjee_xi(x, y), chatterjee_xi(y, x))
```

### Documentation

- Use clear, concise language
- Include code examples
- Update README if adding features
- Add docstrings to all public APIs

## Testing

### Writing Tests

Place tests in `tests/` with names like `test_*.py`:

```python
import pytest
import numpy as np
from src.similarity import chatterjee_xi

def test_xi_perfect_correlation():
    """Test Xi on perfect linear relationship."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    xi = chatterjee_xi(x, y)
    assert xi > 0.9, f"Expected xi > 0.9, got {xi}"

def test_xi_independence():
    """Test Xi on independent variables."""
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    xi = chatterjee_xi(x, y)
    assert -0.1 < xi < 0.1, f"Expected xi ≈ 0, got {xi}"
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific file
pytest tests/test_chatterjee_xi.py

# With coverage
pytest tests/ --cov=src

# Verbose
pytest tests/ -v
```

## Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Full error traceback

### Feature Requests

Include:
- Clear description of the feature
- Use cases and motivation
- Example API (if applicable)

## Questions?

- **Issues**: https://github.com/Professor-Hunt/Xi_Correlation/issues
- **Discussions**: https://github.com/Professor-Hunt/Xi_Correlation/discussions

## Code of Conduct

Be respectful and constructive. We're all here to learn and improve the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
