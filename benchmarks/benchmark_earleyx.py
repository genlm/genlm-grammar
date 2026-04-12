#!/usr/bin/env python3
"""Benchmark: prefix parsing runtime — our Earley vs EarleyX (Stolcke).

Compares per-word-position cumulative runtime on the same grammar and
sentences.  EarleyX is called via subprocess (one batch for all sentences);
per-position timing is read from ``## WordTime`` lines on stderr,
emitted by the instrumented EarleyParser.java.
"""

import argparse
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, LogLocator

from genlm.grammar.treebank import TreebankCFG

# Shared helpers from the main benchmark
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import (
    ACL_COL_WIDTH, aggregate_by_position, fit_power_law,
    load_sentences, plot_series,
)

# ── EarleyX paths ──────────────────────────────────────────────────────────
EARLEYX_DIR = Path(os.environ.get("EARLEYX_DIR", str(Path.home() / "earleyx")))
EARLEYX_CP = f"{EARLEYX_DIR / 'classes'}:{EARLEYX_DIR / 'lib'}/*"

# ── Matplotlib: ACL style ──────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})


# ── Runners ────────────────────────────────────────────────────────────────

def _earleyx_chunk(grammar_file, sents, root, tmpdir, chunk_id, timeout):
    """Run EarleyX on one chunk of sentences. Returns [(pos, ms), ...]."""
    in_path = os.path.join(tmpdir, f"input_{chunk_id}.txt")
    out_pfx = os.path.join(tmpdir, f"out_{chunk_id}")
    with open(in_path, "w") as f:
        for sent in sents:
            f.write(" ".join(t.lstrip("_") for t in sent) + "\n")

    r = subprocess.run( # run EarleyX from the CLI.
        ["java", "-cp", EARLEYX_CP, "parser.Main",
         "-in", in_path, "-out", out_pfx, "-grammar", grammar_file,
         "-obj", "prefix", "-root", root, "-normalprob"],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"EarleyX chunk {chunk_id} failed (rc={r.returncode}):\n"
            f"{r.stderr[:2000]}")

    return [
        (int(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"## WordTime \S+ (\d+) ([\d.]+)", r.stderr)
    ]


def run_earleyx(grammar_file, processed_sents, root="ROOT", n_workers=None,
                timeout=3600):
    """Run EarleyX on preprocessed sentences (parallel across N JVMs).

    Returns list of (word_position, cumulative_ms) parsed from ## WordTime.
    """
    n_workers = min(n_workers or multiprocessing.cpu_count(),
                    len(processed_sents))
    chunk_size = -(-len(processed_sents) // n_workers)  # ceil division
    chunks = [processed_sents[i:i + chunk_size]
              for i in range(0, len(processed_sents), chunk_size)]

    all_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {
                pool.submit(_earleyx_chunk, grammar_file, chunk,
                            root, tmpdir, i, timeout): i
                for i, chunk in enumerate(chunks)
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                all_results.extend(fut.result())
                print(f"\r  {done}/{len(chunks)} chunks", end="", flush=True)
    print()
    return all_results


_worker_parser = None


def _init_worker(grammar_file, backend):
    """Initialize a parser in each worker process."""
    global _worker_parser
    cfg = TreebankCFG.from_string(Path(grammar_file).read_text())
    grammar = cfg.prefix_grammar
    if backend == "rust":
        from genlm.grammar.parse.earley_rust import EarleyRust
        _worker_parser = EarleyRust(grammar)
    elif backend == "rust-rescaled":
        from genlm.grammar.parse.earley_rescaled_rust import EarleyRescaledRust
        _worker_parser = EarleyRescaledRust(grammar)
    else:
        from genlm.grammar.parse.earley import Earley
        _worker_parser = Earley(grammar)


def _process_sentence(sent):
    """Process one sentence, return list of (position, cumulative_ms)."""
    _worker_parser._chart.clear()
    results = []
    t0 = time.time()
    for j in range(1, len(sent) + 1):
        _worker_parser(tuple(sent[:j]))
        results.append((j, (time.time() - t0) * 1000))
    return results


def run_ours(grammar_file, processed_sents, backend="rust", n_workers=None):
    """Run our prefix parser on preprocessed sentences (multiprocessing).

    Returns list of (word_position, cumulative_ms).
    """
    n_workers = n_workers or multiprocessing.cpu_count()
    all_results = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(grammar_file, backend),
    ) as pool:
        futures = {pool.submit(_process_sentence, s): i
                   for i, s in enumerate(processed_sents)}
        done = 0
        for fut in as_completed(futures):
            done += 1
            all_results.extend(fut.result())
            print(f"\r  {done}/{len(processed_sents)}", end="", flush=True)
    print()
    return all_results


# ── Plotting ───────────────────────────────────────────────────────────────

COLORS = {
    "earleyx": "C1",
    "rust": "C0",
    "python": "C2",
    "rust-rescaled": "C3",
}

LABELS = {
    "earleyx": "EarleyX (Java)",
    "rust": "Prefix grammar (Rust)",
    "python": "Prefix grammar (Python)",
    "rust-rescaled": "Prefix grammar (Rust, rescaled)",
}


def setup_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("String Length (Terminals)")
    ax.set_ylabel("Runtime (ms)")
    ax.grid(True, alpha=0.2, which="both", linewidth=0.5)
    ax.xaxis.set_major_locator(FixedLocator(list(range(5, 55, 5))))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())

    def int_fmt(x, _):
        return f"${int(x)}$" if x >= 1 and x == int(x) else f"${x:g}$"

    def sci_fmt(x, _):
        if x <= 0:
            return "$0$"
        exp = int(np.floor(np.log10(x)))
        coeff = x / 10 ** exp
        if abs(coeff - 1.0) < 0.01:
            return f"$10^{{{exp}}}$"
        return f"${coeff:.0f}" + r"\times " + f"10^{{{exp}}}$"

    ax.xaxis.set_major_formatter(FuncFormatter(int_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark: prefix parsing runtime — ours vs EarleyX")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--max-sentences", type=int)
    ap.add_argument("--min-length", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=200)
    ap.add_argument("--min-position", type=int, default=5)
    ap.add_argument("--max-position", type=int, default=50)
    ap.add_argument("--backends", nargs="+", default=["rust"],
                    choices=["python", "rust", "rust-rescaled"])
    ap.add_argument("--n-workers", type=int)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="Per-chunk timeout for EarleyX in seconds (default: 3600)")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--skip-earleyx", action="store_true")
    args = ap.parse_args()

    # ── Load and preprocess ──────────────────────────────────────────────
    cfg = TreebankCFG.from_string(Path(args.grammar).read_text())
    raw = load_sentences(args.sentences, args.max_sentences,
                         args.min_length, args.max_length)
    processed = [cfg.replace_unknown(s)[0] for s in raw]
    print(f"Grammar: {Path(args.grammar).name}  |  Sentences: {len(processed)}")

    # ── Run benchmarks ───────────────────────────────────────────────────
    series = {}

    for backend in args.backends:
        key = backend
        print(f"{LABELS[key]} ...")
        data = run_ours(args.grammar, processed, backend, args.n_workers)
        series[key] = data
        print(f"  {len(data)} data points")

    if not args.skip_earleyx:
        assert (EARLEYX_DIR / "classes" / "parser" / "Main.class").exists(), \
            f"EarleyX not compiled — run 'ant compile' in {EARLEYX_DIR}"
        print(f"{LABELS['earleyx']} ...")
        data = run_earleyx(args.grammar, processed, str(cfg.S),
                           args.n_workers, args.timeout)
        series["earleyx"] = data
        print(f"  {len(data)} data points")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, ACL_COL_WIDTH * 0.75))
    setup_axes(ax)

    for key, data in series.items():
        agg = aggregate_by_position(data)
        agg = agg[(agg["position"] >= args.min_position) &
                  (agg["position"] <= args.max_position)]
        reg = fit_power_law(agg)
        plot_series(agg, reg, ax, LABELS[key], COLORS[key])
        if reg:
            print(f"{LABELS[key]:35s}: b={reg['b']:.2f}, "
                  f"a={reg['a']:.2f}, R²={reg['r2']:.4f}")

    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")
    fig.tight_layout(pad=0.3)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    name = Path(args.grammar).stem
    out = out_dir / f"earleyx_comparison_{name}_{ts}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
