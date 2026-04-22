#!/usr/bin/env python3
"""
Benchmark: prefix parsing on a single long sentence (sparse grammar).

Replicates the experiment from Luong et al. (TACL 2013), Figure 4:
incremental prefix parsing on a discourse grammar where a single
concatenated transcript is parsed word-by-word.

Usage:
    python benchmarks/benchmark_sparse.py \
        --grammar earleyx/grammars/socialDiscourse.grammar \
        --sentence earleyx/data/socialall.ortho.yld.concat \
        --start Discourse \
        --max-position 2000
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import statsmodels.api as sm

from genlm.grammar.treebank import TreebankCFG

# ── EarleyX ───────────────────────────────────────────────────────────────
EARLEYX_DIR = Path(os.environ.get(
    "EARLEYX_DIR", str(Path(__file__).resolve().parent.parent / "earleyx")))
EARLEYX_CP = f"{EARLEYX_DIR / 'classes'}:{EARLEYX_DIR / 'lib'}/*"

# ── Matplotlib: ACL style ─────────────────────────────────────────────────
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
ACL_COL_WIDTH = 3.25


# ── Power-law fit ─────────────────────────────────────────────────────────

def fit_power_law(positions, times):
    """Fit y = a * x^b via log-log OLS. Returns dict or None."""
    valid = [(p, t) for p, t in zip(positions, times) if p > 0 and t > 0]
    if len(valid) < 2:
        return None
    ps, ts = zip(*valid)
    x_log = np.log(np.array(ps, dtype=float))
    y_log = np.log(np.array(ts, dtype=float))
    X = sm.add_constant(x_log)
    res = sm.OLS(y_log, X).fit()
    intercept, slope = res.params
    x_fit = np.linspace(min(ps), max(ps), 200)
    return {
        "b": slope,
        "a": np.exp(intercept),
        "r2": res.rsquared,
        "x_fit": x_fit,
        "y_fit": np.exp(intercept) * x_fit ** slope,
    }


# ── Runners ───────────────────────────────────────────────────────────────

def run_ours(grammar_file, sent, start, backend, max_position):
    """Run our prefix parser on a single long sentence.

    Returns list of (position, cumulative_ms).
    """
    cfg = TreebankCFG.from_string(
        Path(grammar_file).read_text(), start=start)
    grammar = cfg.prefix_grammar

    if backend == "rust":
        from genlm.grammar.parse.earley_rust import EarleyRust
        parser = EarleyRust(grammar)
    elif backend == "rust-rescaled":
        from genlm.grammar.parse.earley_rescaled_rust import EarleyRescaledRust
        parser = EarleyRescaledRust(grammar)
    else:
        from genlm.grammar.parse.earley import Earley
        parser = Earley(grammar)

    n = min(len(sent), max_position)
    results = []
    t0 = time.time()
    for j in range(1, n + 1):
        parser(tuple(sent[:j]))
        results.append((j, (time.time() - t0) * 1000))
        if j % 100 == 0:
            print(f"\r  {j}/{n}", end="", flush=True)
    print()
    return results


def run_earleyx(grammar_file, sent, start, max_position, timeout, extra_flags=()):
    """Run EarleyX on a single long sentence.

    Returns list of (position, cumulative_ms) from ## WordTime lines.

    ``extra_flags`` is a tuple of additional CLI arguments passed through to
    ``parser.Main``. Useful for enabling modes that populate the
    ``completedEdges`` structure the sparse parser depends on:

      - ``("-io", "em", "-maxiteration", "1")``: triggers the inside-outside
        branch (insideOutsideOpt>0), which initialises completedEdges and
        runs the forward pass once, emitting ## WordTime during that pass.
      - ``("-decode", "marginal")``: sets decodeOpt=2, which also initialises
        completedEdges, while keeping the standard parseSentences flow.

    With no extra flags on a sparse grammar, EarleyX's -sparse path will
    either NPE or silently miss completions — see
    EarleyParser.java:1019, where fastChartComplete reads completedEdges
    that were never populated under bare -sparse -obj prefix.
    """
    n = min(len(sent), max_position)
    # EarleyX expects raw words (without _ prefix)
    raw_tokens = [t.lstrip("_") for t in sent[:n]]

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.txt")
        out_pfx = os.path.join(tmpdir, "result")

        with open(in_path, "w") as f:
            f.write(" ".join(raw_tokens) + "\n")

        cmd = [
            "java", "-Xmx4g", "-cp", EARLEYX_CP, "parser.Main",
            "-in", in_path, "-out", out_pfx,
            "-grammar", grammar_file,
            "-obj", "prefix", "-root", start, "-normalprob", "-sparse",
            *extra_flags,
        ]

        print(f"  Running EarleyX on {n} tokens ...")
        print(f"  cmd: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(
                f"EarleyX failed (rc={r.returncode}):\n{r.stderr[:2000]}")

        return [
            (int(m.group(1)), float(m.group(2)))
            for m in re.finditer(r"## WordTime \S+ (\d+) ([\d.]+)", r.stderr)
        ]


# ── Plotting ──────────────────────────────────────────────────────────────

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

    from matplotlib.ticker import FixedLocator
    ax.xaxis.set_major_locator(FixedLocator([50, 100, 500, 1000, 2000]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(int_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_series(positions, times, reg, ax, label, color):
    fit_str = (f" ($a={reg['a']:.2f},\\; b={reg['b']:.2f}$)"
               if reg else "")
    ax.plot(positions, times, "-", lw=1, color=color,
            label=label + fit_str, zorder=10)
    if reg:
        ax.plot(reg["x_fit"], reg["y_fit"], "--", color=color, lw=1, alpha=0.6)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark: prefix parsing on a single long sentence (sparse grammar)")
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--sentence", required=True,
                    help="File with one sentence per line; lines are concatenated")
    ap.add_argument("--start", default="Discourse",
                    help="Start symbol (default: Discourse)")
    ap.add_argument("--max-position", type=int, default=2000)
    ap.add_argument("--min-position", type=int, default=10)
    ap.add_argument("--backends", nargs="+", default=["rust"],
                    choices=["python", "rust", "rust-rescaled"])
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--skip-earleyx", action="store_true")
    ap.add_argument("--earleyx-extra", default="",
                    help="Extra flags passed through to parser.Main, e.g. "
                    "'-io em -maxiteration 1' to force completedEdges init "
                    "via the inside-outside branch, or '-decode marginal' "
                    "for the same effect via decodeOpt=2.")
    args = ap.parse_args()
    earleyx_extra_flags = tuple(args.earleyx_extra.split()) if args.earleyx_extra else ()

    # ── Load sentence ─────────────────────────────────────────────────────
    with open(args.sentence) as f:
        raw_tokens = f.read().split()
    sent = ["_" + t for t in raw_tokens]

    n = min(len(sent), args.max_position)
    print(f"Grammar: {Path(args.grammar).name}")
    print(f"Sentence: {len(raw_tokens)} tokens, benchmarking up to {n}")

    # ── Run benchmarks ────────────────────────────────────────────────────
    series = {}

    for backend in args.backends:
        print(f"{LABELS[backend]} ...")
        data = run_ours(args.grammar, sent, args.start, backend, args.max_position)
        series[backend] = data
        print(f"  {len(data)} data points")

    if not args.skip_earleyx:
        assert (EARLEYX_DIR / "classes" / "parser" / "Main.class").exists(), \
            f"EarleyX not compiled — run 'ant compile' in {EARLEYX_DIR}"
        print(f"{LABELS['earleyx']} ...")
        data = run_earleyx(args.grammar, sent, args.start,
                           args.max_position, args.timeout,
                           extra_flags=earleyx_extra_flags)
        series["earleyx"] = data
        print(f"  {len(data)} data points")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(ACL_COL_WIDTH, ACL_COL_WIDTH * 0.75))
    setup_axes(ax)

    for key, data in series.items():
        positions = [p for p, t in data if args.min_position <= p <= args.max_position]
        times = [t for p, t in data if args.min_position <= p <= args.max_position]
        reg = fit_power_law(positions, times)
        plot_series(positions, times, reg, ax, LABELS[key], COLORS[key])
        if reg:
            print(f"{LABELS[key]:35s}: b={reg['b']:.2f}, "
                  f"a={reg['a']:.2f}, R²={reg['r2']:.4f}")

    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")
    fig.tight_layout(pad=0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    name = Path(args.grammar).stem
    out = out_dir / f"benchmark_sparse_{name}_{ts}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
