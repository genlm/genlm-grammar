# Prefix Parsing is Just Parsing — ACL 2026

Reference implementation and benchmarks for:

> **Prefix Parsing is Just Parsing**
> Clemente Pasti, Andreas Opedal, Timothy J. O'Donnell, Ryan Cotterell, Tim Vieira.
> *Proceedings of ACL, 2026.*

This repository is a **frozen snapshot** of the `genlm-grammar` library as used in the experiments in the paper. It contains the prefix grammar transformation, the Earley
and CKY parsers, the next-token algorithm, and the code needed to reproduce
every experiment reported in the paper.

> For the actively maintained library, in particular if you want to use our parser,
> we recommend to use the parent repo, which is actively mantained.
> **[genlm/genlm-grammar](https://github.com/genlm/genlm-grammar)**
> (on PyPI as `genlm-grammar`). This snapshot is pinned and will not receive
> updates.

## Setup

Clone with submodules (the vendored EarleyX baseline is a submodule):

```bash
git clone --recurse-submodules git@github.com:genlm/prefix-acl-26.git
cd prefix-acl-26
```

Install the Python package and Rust extension in one step:

```bash
pip install .[bench]
```

This invokes `cargo` under the hood via `setuptools-rust` to build the
`genlm_earley` PyO3 extension. A working Rust toolchain (`rustc`, `cargo`)
is required; install via [rustup](https://rustup.rs/) if needed.

> Editable installs (`pip install -e .`) place the compiled extension at the
> repo root, which is not reliably found by multiprocessing workers. The
> benchmarks use a process pool, so a regular `pip install .` is
> recommended.

Build the EarleyX baseline (Java, requires `ant` and a JDK):

```bash
(cd earleyx && ant compile)
```

## Reproducing the figures

### Fig. 1 (main text) — parsing vs. prefix parsing vs. next-token

```bash
python benchmarks/benchmark.py \
    --grammar grammars/grammar_5000.txt \
    --sentences grammars/sentences_500.txt \
    --backend rust-rescaled
```

Produces `results/benchmark_rust-rescaled_grammar_5000_<timestamp>.png`,
which appears in the paper as `tables/next_token_rescaled.png`.

The grammar-size numbers quoted in the caption (116.6k rules for the
original grammar, 333.7k for the prefix grammar) are computed by:

```bash
python benchmarks/benchmark_grammar_size.py \
    --grammar grammars/grammar_5000.txt --start ROOT
```

### Appendix — comparison against EarleyX

```bash
python benchmarks/benchmark_earleyx.py \
    --grammar grammars/grammar_5000.txt \
    --sentences grammars/sentences_500.txt
```

Produces `earleyx_comparison_grammar_5000_<timestamp>.png`
(= `tables/benchmark_earleyx_comparison.png` in the paper).

### Appendix — sparse grammar (Luong 2013 Fig. 4 setup)

```bash
python benchmarks/benchmark_sparse.py \
    --grammar earleyx/grammars/socialDiscourse.grammar \
    --sentence earleyx/data/socialall.ortho.yld.concat \
    --start Discourse --max-position 2000
```

Produces `benchmark_sparse_socialDiscourse_<timestamp>.png`
(= `tables/sparse.png` in the paper).

## Modifications to EarleyX

The `earleyx` submodule tracks a single-commit fork of
[`lmthang/earleyx`](https://github.com/lmthang/earleyx). The change is
**instrumentation only** — the parsing algorithm and its numerical outputs
are unchanged, the only purpose of teh commit is to track incremental, per-token
runtime.

**Commit `1171b93`** — *Add per-word timing instrumentation for benchmarking.*
Touches `src/parser/EarleyParser.java` (+4 / -2): emits per-word cumulative
wall-clock time so the benchmark harness can plot log–log runtime curves
without conflating parsing time with grammar-loading and I/O.

To verify the delta:

```bash
git -C earleyx diff upstream/master..HEAD -- src/parser/EarleyParser.java
```

## Citation

```bibtex
@inproceedings{pasti2026prefix,
  title     = {Prefix Parsing is Just Parsing},
  author    = {Pasti, Clemente and Opedal, Andreas and
               O'Donnell, Timothy J. and Cotterell, Ryan and
               Vieira, Tim},
  booktitle = {Proceedings of ACL},
  year      = {2026},
}
```

## License

The `genlm-grammar` code is licensed under the terms in `LICENSE`. The
vendored EarleyX fork retains its original license in `earleyx/`.
