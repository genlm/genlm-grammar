#!/usr/bin/env python3
"""
Benchmark: Compare regular parsing, prefix parsing, and next-token weights.

Supports both Python and Rust Earley backends.
Produces publication-quality plots sized for a single ACL column.
"""
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

from genlm.grammar.treebank import TreebankCFG

# ── Matplotlib: LaTeX / Computer Modern for ACL figures ──────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

# ACL single-column width
ACL_COL_WIDTH = 3.25  # inches

# ── Worker state (one grammar + parser per process) ──────────────────────────
_worker_cfg = None
_worker_parser = None
_worker_backend = None


def load_grammar(path):
    with open(path) as f:
        return TreebankCFG.from_string(f.read())


def load_sentences(path, max_sentences=None, min_length=3, max_length=100):
    sentences = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_sentences and i >= max_sentences:
                break
            tokens = line.strip().split()
            if min_length <= len(tokens) <= max_length:
                sentences.append(tokens)
    return sentences


def init_worker(grammar_file, use_prefix_grammar, backend):
    global _worker_cfg, _worker_parser, _worker_backend
    _worker_cfg = load_grammar(grammar_file)
    grammar = _worker_cfg.prefix_grammar if use_prefix_grammar else _worker_cfg
    _worker_backend = backend
    if backend == "rust":
        from genlm.grammar.parse.earley_rust import EarleyRust
        _worker_parser = EarleyRust(grammar)
    elif backend == "rust-rescaled":
        from genlm.grammar.parse.earley_rescaled_rust import EarleyRescaledRust
        _worker_parser = EarleyRescaledRust(grammar)
    elif backend == "python-rescaled":
        from genlm.grammar.parse.earley_rescaled import Earley
        _worker_parser = Earley(grammar)
    else:
        from genlm.grammar.parse.earley import Earley
        _worker_parser = Earley(grammar)


# ── Underflow tracking ───────────────────────────────────────────────────────
# We track the minimum nonzero weight seen at any prefix position to detect
# whether the computation is approaching float64 underflow (~5e-324 denorm,
# ~2.2e-308 normal).
_FLOAT64_MIN_NORMAL = np.finfo(np.float64).tiny  # ~2.2e-308


def _check_weight(w, sent_idx, pos, kind):
    """Return a warning string if w is zero or subnormal, else None."""
    if isinstance(w, (int, float)):
        v = float(w)
    else:
        v = float(w)
    if v == 0.0:
        return f"UNDERFLOW: {kind} weight is exactly 0 at sent={sent_idx} pos={pos}"
    if 0 < v < _FLOAT64_MIN_NORMAL:
        return f"SUBNORMAL: {kind} weight={v:.4e} at sent={sent_idx} pos={pos}"
    return None


def process_regular(args):
    sent_idx, sentence = args
    global _worker_cfg, _worker_parser
    sent, _ = _worker_cfg.replace_unknown(sentence)
    _worker_parser._chart.clear()
    results = []
    underflow_warnings = []
    start = time.time()
    for j in range(1, len(sent) + 1):
        w = _worker_parser(tuple(sent[:j]))
        elapsed = (time.time() - start) * 1000
        results.append((j, elapsed))
        warn = _check_weight(w, sent_idx, j, "regular")
        if warn:
            underflow_warnings.append(warn)
    return results, underflow_warnings


def process_prefix(args):
    sent_idx, sentence = args
    global _worker_cfg, _worker_parser
    sent, _ = _worker_cfg.replace_unknown(sentence)
    _worker_parser._chart.clear()
    results = []
    underflow_warnings = []
    start = time.time()
    for j in range(1, len(sent) + 1):
        w = _worker_parser(tuple(sent[:j]))
        elapsed = (time.time() - start) * 1000
        results.append((j, elapsed))
        warn = _check_weight(w, sent_idx, j, "prefix")
        if warn:
            underflow_warnings.append(warn)
    return results, underflow_warnings


def process_next_token(args):
    sent_idx, sentence = args
    global _worker_cfg, _worker_parser
    sent, _ = _worker_cfg.replace_unknown(sentence)
    _worker_parser._chart.clear()
    results = []
    underflow_warnings = []
    start = time.time()
    for j in range(1, len(sent) + 1):
        cols = _worker_parser.chart(tuple(sent[:j]))
        ntw = _worker_parser.next_token_weights(cols)
        elapsed = (time.time() - start) * 1000
        results.append((j, elapsed))
        # Check all next-token weights for underflow
        for sym, w in ntw.items():
            warn = _check_weight(w, sent_idx, j, f"next_token[{sym}]")
            if warn:
                underflow_warnings.append(warn)
    return results, underflow_warnings


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(grammar_file, sentences, process_fn, use_prefix_grammar,
                  n_workers, backend):
    n_workers = n_workers or multiprocessing.cpu_count()
    all_results = []
    all_warnings = []
    indexed_sentences = list(enumerate(sentences))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(grammar_file, use_prefix_grammar, backend),
    ) as executor:
        futures = {
            executor.submit(process_fn, item): i
            for i, item in enumerate(indexed_sentences)
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                results, warns = future.result()
                all_results.extend(results)
                all_warnings.extend(warns)
            except Exception as e:
                print(f"Error: {e}")
            print(f"\r  {done}/{len(sentences)}", end="", flush=True)
    print()
    return all_results, all_warnings


# ── Statistics & plotting ────────────────────────────────────────────────────

def aggregate_by_position(results):
    """Aggregate raw (position, time_ms) pairs into per-position statistics."""
    df = pd.DataFrame(results, columns=["position", "time_ms"])
    agg = df.groupby("position")["time_ms"].agg(["mean", "std", "count"])
    agg["std"] = agg["std"].fillna(0)
    margin = np.where(
        agg["count"] > 1,
        stats.t.ppf(0.975, agg["count"] - 1) * agg["std"] / np.sqrt(agg["count"]),
        0,
    )
    agg["ci_lo"] = agg["mean"] - margin
    agg["ci_hi"] = agg["mean"] + margin
    return agg.reset_index()


def fit_power_law(agg):
    """Fit y = a * x^b via weighted log-log regression. Input: aggregated df."""
    valid = agg[(agg["position"] > 0) & (agg["mean"] > 0)]
    if len(valid) < 2:
        return None
    x_log = np.log(valid["position"].values)
    y_log = np.log(valid["mean"].values)
    X = sm.add_constant(x_log)
    res = sm.WLS(y_log, X, weights=valid["count"].values).fit()
    intercept, slope = res.params
    x_fit = np.linspace(valid["position"].min(), valid["position"].max(), 100)
    return {
        "b": slope,
        "a": np.exp(intercept),
        "r2": res.rsquared,
        "x_fit": x_fit,
        "y_fit": np.exp(intercept) * x_fit ** slope,
    }


def plot_series(agg, reg, ax, label, color):
    """Plot aggregated time series with CI band and optional regression fit."""
    if agg.empty:
        return
    fit_str = f" (${{a}}\!=\!{reg['a']:.2f},\\; {{b}}\!=\!{reg['b']:.2f}$)" if reg else ""
    ax.plot(agg["position"], agg["mean"], "o-", ms=1.5, lw=1, color=color,
            label=label + fit_str, zorder=10)
    ax.fill_between(agg["position"], agg["ci_lo"], agg["ci_hi"],
                    alpha=0.15, color=color)
    if reg:
        ax.plot(reg["x_fit"], reg["y_fit"], "--", color=color, lw=1, alpha=0.6)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark regular/prefix/next-token parsing."
    )
    parser.add_argument("--grammar", required=True)
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--n-workers", type=int)
    parser.add_argument("--backend", choices=["python", "rust", "rust-rescaled", "python-rescaled"], default="rust")
    parser.add_argument("--skip-regular", action="store_true")
    parser.add_argument("--skip-prefix", action="store_true")
    parser.add_argument("--skip-next-token", action="store_true")
    parser.add_argument("--min-position", type=int, default=10)
    parser.add_argument("--max-position", type=int, default=50)
    args = parser.parse_args()

    sentences = load_sentences(args.sentences, args.max_sentences,
                               args.min_length, args.max_length)
    print(f"Backend: {args.backend}")
    print(f"Sentences: {len(sentences)}")
    grammar_name = Path(args.grammar).stem

    results = {}
    all_warnings = []

    if not args.skip_regular:
        print("Regular parsing...")
        res, warns = run_benchmark(
            args.grammar, sentences, process_regular, False,
            args.n_workers, args.backend)
        results["regular"] = res
        all_warnings.extend(warns)

    if not args.skip_prefix:
        print("Prefix parsing...")
        res, warns = run_benchmark(
            args.grammar, sentences, process_prefix, True,
            args.n_workers, args.backend)
        results["prefix"] = res
        all_warnings.extend(warns)

    if not args.skip_next_token:
        print("Next token weights...")
        res, warns = run_benchmark(
            args.grammar, sentences, process_next_token, True,
            args.n_workers, args.backend)
        results["next_token"] = res
        all_warnings.extend(warns)

    # ── Report underflow warnings ────────────────────────────────────────
    if all_warnings:
        print(f"\n{'='*60}")
        print(f"UNDERFLOW/SUBNORMAL WARNINGS: {len(all_warnings)}")
        print(f"{'='*60}")
        # Print unique warnings (deduplicate by prefix)
        seen = set()
        for w in all_warnings:
            key = w.split(" at ")[0]
            if key not in seen:
                print(f"  {w}")
                seen.add(key)
            if len(seen) >= 20:
                print(f"  ... and {len(all_warnings) - 20} more")
                break
        print(f"{'='*60}\n")
    else:
        print("\nNo underflow detected.\n")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, ACL_COL_WIDTH * 0.75))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("String Length (Terminals)")
    ax.set_ylabel("Runtime (ms)")
    ax.grid(True, alpha=0.2, which="both", linewidth=0.5)

    # Explicit tick positions for clean log-scale labels
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, LogLocator
    ax.xaxis.set_major_locator(FixedLocator(list(range(5, 55, 5))))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())

    def _int_formatter(x, _):
        if x >= 1 and x == int(x):
            return f"${int(x)}$"
        return f"${x:g}$"

    def _sci_formatter(x, _):
        if x <= 0:
            return "$0$"
        exp = int(np.floor(np.log10(x)))
        coeff = x / 10 ** exp
        if abs(coeff - 1.0) < 0.01:
            return f"$10^{{{exp}}}$"
        return f"${coeff:.0f}" + r"\times " + f"10^{{{exp}}}$"

    ax.xaxis.set_major_formatter(FuncFormatter(_int_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_sci_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())

    colors = {"regular": "C3", "prefix": "C0", "next_token": "C2"}
    labels = {
        "regular": "Parsing",
        "prefix": "Prefix parsing",
        "next_token": "Next-token weights",
    }

    for name, data in results.items():
        agg = aggregate_by_position(data)
        agg = agg[(agg["position"] >= args.min_position) &
                  (agg["position"] <= args.max_position)]
        reg = fit_power_law(agg)
        plot_series(agg, reg, ax, labels[name], colors[name])
        if reg:
            print(f"{labels[name]:25s}: b={reg['b']:.2f}, "
                  f"a={reg['a']:.2f}, R²={reg['r2']:.2f}")

    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")

    fig.tight_layout(pad=0)

    # Save with timestamp
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    outfile = out_dir / f"benchmark_{args.backend}_{grammar_name}_{timestamp}.png"
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    print(f"Saved: {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    main()
