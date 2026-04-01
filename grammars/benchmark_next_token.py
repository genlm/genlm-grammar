#!/usr/bin/env python3
"""Benchmark: Compare regular parsing, prefix parsing, and next token weights."""
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
import statsmodels.api as sm

from genlm.grammar.treebank import TreebankCFG
from genlm.grammar.parse.earley import Earley

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times']

_worker_cfg = None
_worker_parser = None


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


def init_worker(grammar_file, use_prefix_grammar):
    global _worker_cfg, _worker_parser
    _worker_cfg = load_grammar(grammar_file)
    grammar = _worker_cfg.prefix_grammar if use_prefix_grammar else _worker_cfg
    _worker_parser = Earley(grammar)


def process_regular(sentence):
    global _worker_cfg, _worker_parser
    sent, _ = _worker_cfg.replace_unknown(sentence)
    _worker_parser._chart.clear()
    results = []
    start = time.time()
    for j in range(1, len(sent) + 1):
        _worker_parser(tuple(sent[:j]))
        results.append((j, (time.time() - start) * 1000))
    return results


def process_prefix(sentence):
    global _worker_cfg, _worker_parser
    sent, _ = _worker_cfg.replace_unknown(sentence)
    _worker_parser._chart.clear()
    results = []
    start = time.time()
    for j in range(1, len(sent) + 1):
        _worker_parser(tuple(sent[:j]))
        results.append((j, (time.time() - start) * 1000))
    return results


def process_next_token(sentence):
    global _worker_cfg, _worker_parser
    sent, _ = _worker_cfg.replace_unknown(sentence)
    _worker_parser._chart.clear()
    results = []
    start = time.time()
    for j in range(1, len(sent) + 1):
        cols = _worker_parser.chart(tuple(sent[:j]))
        _worker_parser.next_token_weights(cols)
        results.append((j, (time.time() - start) * 1000))
    return results


def run_benchmark(grammar_file, sentences, process_fn, use_prefix_grammar, n_workers):
    n_workers = n_workers or multiprocessing.cpu_count()
    all_results = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(grammar_file, use_prefix_grammar)
    ) as executor:
        futures = [executor.submit(process_fn, s) for s in sentences]
        for i, future in enumerate(as_completed(futures)):
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"Error: {e}")
            print(f"\r{i+1}/{len(sentences)}", end='', flush=True)
    print()
    return all_results


def compute_regression(lengths, means, counts=None, min_length=None, max_length=None):
    """Compute WLS regression in log-log space, weighted by sample counts."""
    lengths, means = np.array(lengths), np.array(means)
    if counts is not None:
        counts = np.array(counts)
    mask = (lengths > 0) & (means > 0)
    if min_length:
        mask &= (lengths >= min_length)
    if max_length:
        mask &= (lengths <= max_length)
    if mask.sum() < 2:
        return None
    x_log, y_log = np.log(lengths[mask]), np.log(means[mask])
    weights = counts[mask] if counts is not None else np.ones(len(x_log))
    
    # WLS regression: weight by number of samples at each length
    X = sm.add_constant(x_log)
    model = sm.WLS(y_log, X, weights=weights)
    results = model.fit()
    intercept, slope = results.params
    
    x_fit = np.linspace(lengths[mask].min(), lengths[mask].max(), 100)
    return {
        'b': slope, 'a': np.exp(intercept), 'r2': results.rsquared,
        'x_fit': x_fit, 'y_fit': np.exp(intercept) * x_fit**slope
    }


def plot_results(results, ax, label, color, min_length_regression=None, max_length_regression=None):
    if not results:
        return None
    lengths = np.array([l for l, _ in results])
    times = np.array([t for _, t in results])
    
    unique = sorted(set(lengths))
    means, ci_lo, ci_hi, counts = [], [], [], []
    for l in unique:
        t = times[lengths == l]
        m, s, n = np.mean(t), np.std(t, ddof=1) if len(t) > 1 else 0, len(t)
        margin = stats.t.ppf(0.975, n-1) * s / np.sqrt(n) if n > 1 else 0
        means.append(m)
        ci_lo.append(m - margin)
        ci_hi.append(m + margin)
        counts.append(n)
    
    unique, means, ci_lo, ci_hi, counts = np.array(unique), np.array(means), np.array(ci_lo), np.array(ci_hi), np.array(counts)
    
    # Filter to regression span
    mask = np.ones(len(unique), dtype=bool)
    if min_length_regression:
        mask &= (unique >= min_length_regression)
    if max_length_regression:
        mask &= (unique <= max_length_regression)
    
    ax.plot(unique[mask], means[mask], 'o-', ms=3, lw=2, color=color, label=label, zorder=10)
    ax.fill_between(unique[mask], ci_lo[mask], ci_hi[mask], alpha=0.15, color=color)
    
    reg = compute_regression(unique, means, counts, min_length_regression, max_length_regression)
    if reg:
        ax.plot(reg['x_fit'], reg['y_fit'], '--', color=color, lw=2, alpha=0.6,
                label=f"{label} (a={reg['a']:.2e}, b={reg['b']:.2f}, R²={reg['r2']:.3f})")
    return reg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", required=True)
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--output", default="benchmark.png")
    parser.add_argument("--n-workers", type=int)
    parser.add_argument("--skip-regular", action="store_true")
    parser.add_argument("--skip-prefix", action="store_true")
    parser.add_argument("--skip-next-token", action="store_true")
    parser.add_argument("--min-length-regression", type=int, default=5)
    parser.add_argument("--max-length-regression", type=int, default=55)
    args = parser.parse_args()

    sentences = load_sentences(args.sentences, args.max_sentences, args.min_length)
    print(f"Sentences: {len(sentences)}")
    
    results = {}
    
    if not args.skip_regular:
        print("Regular parsing...")
        results['regular'] = run_benchmark(args.grammar, sentences, process_regular, False, args.n_workers)
    
    if not args.skip_prefix:
        print("Prefix parsing...")
        results['prefix'] = run_benchmark(args.grammar, sentences, process_prefix, True, args.n_workers)
    
    if not args.skip_next_token:
        print("Next token weights...")
        results['next_token'] = run_benchmark(args.grammar, sentences, process_next_token, True, args.n_workers)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Length (tokens)', fontsize=16)
    ax.set_ylabel('Cumulative Runtime (ms)', fontsize=16)
    ax.grid(True, alpha=0.3, which='both')
    
    colors = {'regular': 'red', 'prefix': 'blue', 'next_token': 'green'}
    labels = {'regular': 'Regular Parsing', 'prefix': 'Prefix Parsing', 'next_token': 'Next Token Weights'}
    
    for name, data in results.items():
        reg = plot_results(data, ax, labels[name], colors[name], args.min_length_regression, args.max_length_regression)
        if reg:
            print(f"{labels[name]}: b={reg['b']:.4f}, a={reg['a']:.4e}, R²={reg['r2']:.4f}")

    ax.legend(fontsize=10, loc='upper left')
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
