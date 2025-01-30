[![Docs](https://github.com/chi-collective/genlm-grammar/actions/workflows/docs.yml/badge.svg)](https://probcomp.github.io/genlm-cfg/)
[![Tests](https://github.com/chi-collective/genlm-grammar/actions/workflows/pytest.yml/badge.svg)](https://github.com/probcomp/genlm-cfg/actions/workflows/pytest.yml)

# GenLM Grammar

A Python library for working with weighted context-free grammars (WCFGs) and finite state automata (FSA). The library provides efficient implementations for grammar operations, parsing algorithms, and language model functionality.

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
git clone git@github.com:chi-collective/genlm-grammar.git
cd genlm_grammar
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
