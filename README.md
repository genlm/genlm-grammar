[![Docs](https://github.com/chisym/genlm-grammar/actions/workflows/docs.yml/badge.svg)](https://chisym.github.io/genlm-grammar/)
[![Tests](https://github.com/chisym/genlm-grammar/actions/workflows/pytest.yml/badge.svg)](https://chisym.com/chisym/genlm-grammar/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/github/chisym/genlm-grammar/graph/badge.svg?token=TQBAQ1uA6y)](https://codecov.io/github/chisym/genlm-grammar)

# GenLM Grammar

A Python library for working with weighted context-free grammars (WCFGs), weighted finite state automata (WFSAs) and weighted finite state transducers (WFSTs). The library provides efficient implementations for grammar operations, parsing algorithms, and language model functionality.

## Key Features

### Grammar Operations
- Support for weighted context-free grammars with various semirings (Boolean, Float, Real, MaxPlus, MaxTimes, etc.)
- Grammar transformations:
  - Local normalization
  - Removal of nullary rules and unary cycles
  - Grammar binarization
  - Length truncation
  - Renaming/renumbering of nonterminals

### Parsing Algorithms
- Earley parsing (O(n³|G|) complexity)
  - Standard implementation
  - Rescaled version for numerical stability
- CKY parsing
  - Incremental CKY with chart caching
  - Support for prefix computations

### Language Model Interface
- `BoolCFGLM`: Boolean-weighted CFG language model
- `CKYLM`: Probabilistic CFG language model using CKY
- `EarleyLM`: Language model using Earley parsing

### Finite State Automata
- Weighted FSA implementation
- Operations:
  - Epsilon removal
  - Minimization (Brzozowski's algorithm)
  - Determinization
  - Composition
  - Reversal
  - Kleene star/plus

### Additional Features
- Semiring abstractions (Boolean, Float, Log, Entropy, etc.)
- Efficient chart and agenda-based algorithms
- Grammar-FST composition
- Visualization support via Graphviz

## Quick Start

### Installation

Clone the repository:

```bash
git clone git@github.com:chisym/genlm-grammar.git
cd genlm-grammar
```

and install with pip:

```bash
pip install .
```

This installs the package without development dependencies. For development, install in editable mode with:

```bash
pip install -e ".[test,docs]"
```
which also installs the dependencies needed for testing (test) and documentation (docs).

## Requirements

- Python >= 3.10
- The core dependencies listed in the `setup.py` file of the repository.

## Testing

When test dependencies are installed, the test suite can be run via:
```bash
pytest tests
```

## Documentation

Documentation is generated using [mkdocs](https://www.mkdocs.org/) and hosted on GitHub Pages. To build the documentation, run:

```bash
mkdocs build
```

To serve the documentation locally, run:

```bash
mkdocs serve
```
