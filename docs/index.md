# GenLM Grammar Documentation

This is a Python library for working with weighted context-free grammars (WCFGs) and finite state machines (FSAs). It provides implementations of various parsing algorithms and language model capabilities.

## Core Components

### Grammar Types
- [CFG](reference/genlm.grammar/cfg): Context-free grammar implementation with support for:
    - Grammar normalization and transformation
    - Conversion to a character-level grammar

### Language Models
- [LM](reference/genlm.grammar/lm): Base language model class
- [BoolCFGLM](reference/genlm.grammar/cfglm/#genlm.grammar.cfglm.BoolCFGLM): Boolean-weighted CFG language model using Earley or CKY parsing
- [CKYLM](reference/genlm.grammar/parse/cky/#genlm.grammar.parse.cky.CKYLM): CKY-based parsing for weighted CFGs
- [EarleyLM](reference/genlm.grammar/parse/earley_rescaled/#genlm.grammar.parse.earley_rescaled.EarleyLM): Earley-based parsing implementation for weighted CFGs

### Parsing Algorithms
- [Earley Parser](reference/genlm.grammar/parse/earley_rescaled): Earley parsing algorithm with rescaling for numerical stability
- [IncrementalCKY](reference/genlm.grammar/parse/cky): Incremental version of CKY with chart caching

### Finite State Machines
- [FST](reference/genlm.grammar/fst): Weighted finite-state transducer implementation
- [WFSA](reference/genlm.grammar/wfsa/base): Weighted finite-state automaton base class

### Mathematical Components
- [Semiring](reference/genlm.grammar/semiring): Abstract semiring implementations including:
    - Boolean
    - Float
    - Log
    - Expectation
- [Chart](reference/genlm.grammar/chart): Weighted chart data structure with semiring operations
- [WeightedGraph](reference/genlm.grammar/linear): Graph implementation for solving algebraic path problems

### Utilities
- [LarkStuff](reference/genlm.grammar/lark_interface): Interface for converting Lark grammars to genlm-cfg format
- [format_table](reference/genlm.grammar/util): Utility functions for formatting and displaying tables

## Key Features

- Support for various weighted grammar formalisms
- Multiple parsing algorithm implementations
- Efficient chart caching and incremental parsing
- Composition operations between FSTs and CFGs
- Semiring abstractions for different weight types
- Visualization capabilities for debugging and analysis

## Common Operations

### Creating a Grammar
```python
from genlm.grammar.cfg import CFG
from genlm.grammar.semiring import Float

# Create from string representation
cfg = CFG.from_string(grammar_string, semiring=Float)
```

### Using a Language Model
```python
from genlm.grammar.cfglm import BoolCFGLM

# Create language model from genlm.grammar
lm = BoolCFGLM(cfg, alg='earley')  # or alg='cky'

# Get next token weights
probs = lm.p_next(context)
```
