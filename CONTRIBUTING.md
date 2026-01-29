# Contributing to ConsciousnessX

Thank you for your interest in contributing to ConsciousnessX! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [consciousnessx@example.com](mailto:consciousnessx@example.com).

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find that the problem has already been reported. When creating a bug report, please include:

- A descriptive title
- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, package versions)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- A clear title and description
- Explain why this enhancement would be useful
- Provide examples of how the enhancement would be used
- If applicable, provide mockups or examples

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Write tests** for your changes
4. **Update documentation** as needed
5. **Ensure all tests pass** locally
6. **Submit a pull request** with a clear description

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Napiersnotes/consciousnessX.git
cd consciousnessX
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e ".[dev,test]"
```

4. Install pre-commit hooks (optional):
```bash
pre-commit install
```

## Coding Standards

### Python Style

We follow PEP 8 style guidelines with some modifications:

- Line length: 100 characters
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Use descriptive variable names

### Formatting

We use Black for code formatting:

```bash
black src tests
```

### Linting

We use Flake8 for linting:

```bash
flake8 src tests
```

### Type Checking

We use MyPy for type checking:

```bash
mypy src
```

### Import Sorting

We use isort for import sorting:

```bash
isort src tests
```

## Testing

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/unit/core/test_microtubule_simulator.py
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

- Write unit tests for all new functionality
- Test both success and failure cases
- Use pytest fixtures for common test setup
- Aim for high code coverage (>80%)

## Documentation

### Building Documentation

```bash
cd docs
make html
```

The documentation will be in `docs/_build/html/`.

### Writing Documentation

- Use clear, concise language
- Include code examples
- Document all public APIs
- Keep documentation in sync with code changes

## Project Structure

```
consciousnessX/
├── src/                    # Source code
│   ├── core/              # Core consciousness modules
│   ├── training/          # Training functionality
│   ├── evaluation/        # Evaluation metrics
│   ├── hardware/          # Hardware integration
│   ├── virtual_bio/       # Virtual biological systems
│   ├── visualization/     # Visualization tools
│   └── consciousnessx/    # API and web interfaces
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/                 # Documentation
├── configs/              # Configuration files
├── examples/             # Example usage
└── scripts/              # Utility scripts
```

## Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: feat, fix, docs, style, refactor, test, chore

Example:
```
feat(core): add quantum coherence measurement

Add new method to measure quantum coherence in microtubules.
This enables more accurate consciousness quantification.

Closes #123
```

## Release Process

Releases are managed automatically through GitHub Actions. To trigger a release:

1. Update version in `src/__init__.py`
2. Update `CHANGELOG.md`
3. Create a pull request merging to `main`
4. Tag the commit with version number (e.g., `v0.2.0`)
5. GitHub Actions will build and publish the release

## Questions?

For questions about contributing, please open an issue or contact [consciousnessx@example.com](mailto:consciousnessx@example.com).

Thank you for contributing to ConsciousnessX!