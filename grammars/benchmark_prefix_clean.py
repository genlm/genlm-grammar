#!/usr/bin/env python3
"""
Clean benchmark: Compute prefix probabilities and plot runtime on log-log axes.
Tracks runtime for each individual prefix.
PARALLELIZED VERSION
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Use serif font (LaTeX-like) for title and legend
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times']

from genlm.grammar.cfg import CFG
from genlm.grammar.treebank import TreebankCFG
from genlm.grammar.semiring import Float
from genlm.grammar.parse.earley import Earley
from genlm.grammar.cfglm import locally_normalize


def load_grammar(grammar_file):
    with open(grammar_file, 'r') as f:
        grammar_str = f.read()
    return TreebankCFG.from_string(grammar_str)


def load_sentences(sentences_file, max_sentences=None, min_length=3, max_length=100):
    sentences = []
    with open(sentences_file, 'r') as f:
        for i, line in enumerate(f):
            if max_sentences and i >= max_sentences:
                break
            tokens = line.strip().split()
            if len(tokens) >= min_length and len(tokens) <= max_length:  # Filter by minimum length
                sentences.append(tokens)
    return sentences


# Module-level variables for worker processes (loaded once per worker)
_worker_cfg = None
_worker_prefix_parser = None
_worker_regular_parser = None


def _init_worker_prefix(grammar_file):
    """Initialize worker process for prefix parsing - loads grammar ONCE per worker."""
    global _worker_cfg, _worker_prefix_parser
    _worker_cfg = load_grammar(grammar_file)
    prefix_grammar = _worker_cfg.prefix_grammar
    _worker_prefix_parser = Earley(prefix_grammar)
    print(f"Worker initialized: grammar size={_worker_cfg.size}, prefix grammar size={prefix_grammar.size}")


def _init_worker_regular(grammar_file):
    """Initialize worker process for regular parsing - loads grammar ONCE per worker."""
    global _worker_cfg, _worker_regular_parser
    _worker_cfg = load_grammar(grammar_file)
    _worker_regular_parser = Earley(_worker_cfg)
    print(f"Worker initialized: grammar size={_worker_cfg.size}")


def process_sentence_prefix(sentence):
    """
    Process a single sentence for prefix parsing.
    Uses pre-loaded grammar from worker initialization.
    
    Args:
        sentence: List of tokens (sentence)
    
    Returns:
        List of (length, runtime_ms) tuples
    """
    global _worker_cfg, _worker_prefix_parser
    
    results = []
    
    # Handle OOV
    sent_processed, _ = _worker_cfg.replace_unknown(sentence)
    
    # Clear cache
    _worker_prefix_parser._chart.clear()
    
    # Compute each prefix
    start = time.time()
    for j in range(1, len(sent_processed) + 1):
        prefix = tuple(sent_processed[:j])
        weight = _worker_prefix_parser(prefix)
        elapsed = time.time() - start
        
        if elapsed > 0:
            results.append((len(prefix), elapsed * 1000))
    
    return results


def process_sentence_regular(sentence):
    """
    Process a single sentence for regular parsing.
    Uses pre-loaded grammar from worker initialization.
    
    Args:
        sentence: List of tokens (sentence)
    
    Returns:
        List of (length, runtime_ms) tuples
    """
    global _worker_cfg, _worker_regular_parser
    
    results = []
    
    # Handle OOV
    sent_processed, _ = _worker_cfg.replace_unknown(sentence)
    
    # Clear cache
    _worker_regular_parser._chart.clear()
    
    # Compute each prefix
    start = time.time()
    for j in range(1, len(sent_processed) + 1):
        prefix = tuple(sent_processed[:j])
        weight = _worker_regular_parser(prefix)
        elapsed = time.time() - start
        
        if elapsed > 0:
            results.append((len(prefix), elapsed * 1000))
    
    return results


def benchmark_prefix_parsing_parallel(grammar_file, sentences, n_workers=None):
    """
    Parallelized version of benchmark_prefix_parsing.
    Grammar is loaded ONCE per worker process via initializer.
    
    Args:
        grammar_file: Path to grammar file
        sentences: List of sentences to process
        n_workers: Number of parallel workers (default: number of CPUs)
    
    Returns: list of (length, prefix_parse_runtime_ms) tuples
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    all_results = []
    total_tasks = len(sentences)
    
    # Use initializer to load grammar ONCE per worker
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker_prefix,
        initargs=(grammar_file,)
    ) as executor:
        futures = {
            executor.submit(process_sentence_prefix, sentence): i 
            for i, sentence in enumerate(sentences)
        }
        
        completed = 0
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
                completed += 1
                progress = (completed / total_tasks) * 100
                print(f"Progress: {completed}/{total_tasks} ({progress:.1f}%)", end='\r')
            except Exception as e:
                sentence_idx = futures[future]
                print(f"\nError processing sentence {sentence_idx}: {e}")
                completed += 1
    
    print()
    return all_results


def benchmark_parsing_parallel(grammar_file, sentences, n_workers=None):
    """
    Parallelized version of benchmark_parsing.
    Grammar is loaded ONCE per worker process via initializer.
    
    Args:
        grammar_file: Path to grammar file
        sentences: List of sentences to process
        n_workers: Number of parallel workers (default: number of CPUs)
    
    Returns: list of (length, parse_runtime_ms) tuples
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    all_results = []
    total_tasks = len(sentences)
    
    # Use initializer to load grammar ONCE per worker
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker_regular,
        initargs=(grammar_file,)
    ) as executor:
        futures = {
            executor.submit(process_sentence_regular, sentence): i 
            for i, sentence in enumerate(sentences)
        }
        
        completed = 0
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
                completed += 1
                progress = (completed / total_tasks) * 100
                print(f"Progress: {completed}/{total_tasks} ({progress:.1f}%)", end='\r')
            except Exception as e:
                sentence_idx = futures[future]
                print(f"\nError processing sentence {sentence_idx}: {e}")
                completed += 1
    
    print()
    return all_results


def compute_loglog_regression(unique_lengths, means):
    """
    Compute linear regression on log-log scale to estimate power law coefficients.
    
    Fits: log(y) = log(a) + b * log(x)
    Which corresponds to: y = a * x^b
    
    Args:
        unique_lengths: array of x values (lengths)
        means: array of y values (mean runtimes)
    
    Returns:
        tuple: (exponent_b, coefficient_a, r_squared, fitted_line_x, fitted_line_y)
    """
    from scipy import stats
    
    # Convert to numpy arrays
    unique_lengths = np.array(unique_lengths)
    means = np.array(means)
    
    # Filter out zeros and negative values
    mask = (unique_lengths > 0) & (means > 0)
    if np.sum(mask) < 2:
        return None, None, None, None, None
    
    x_log = np.log(unique_lengths[mask])
    y_log = np.log(means[mask])
    
    # Perform linear regression on log-transformed data
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
    
    # Extract coefficients
    exponent_b = slope  # This is b in --> log(y) = log(a) +  log(x)*b
    coefficient_a = np.exp(intercept)  # This is a in --> log(y) = log(a) +  log(x)*b
    
    # Compute R-squared
    r_squared = r_value ** 2
    
    # Generate fitted line for plotting
    x_fit = np.linspace(unique_lengths[mask].min(), unique_lengths[mask].max(), 100)
    y_fit = coefficient_a * (x_fit ** exponent_b)
    
    return exponent_b, coefficient_a, r_squared, x_fit, y_fit


def plot_results(results, output_file="benchmark_prefix_loglog.png", MAX=70, MIN=3,
                 ax=None, label=None, color='blue', scatter_alpha=0.1, 
                 show_regression=True):
    """
    Plot runtime vs prefix length on log-log axes.

    Args:
        results: list of (prefix_length, runtime_ms) tuples
        output_file: output file path (only used if ax is None)
        MAX: maximum length to plot
        MIN: minimum length to plot
        ax: matplotlib axis to plot on (if None, creates new figure)
        label: label for this dataset (for legend)
        color: color for the plot elements
        scatter_alpha: transparency for scatter points (0.0 = fully transparent, 1.0 = opaque)
        show_regression: if True, plot the fitted power law line
        return_coefficients: if True, return regression coefficients
    
    Returns:
        ax (and optionally coefficients dict if return_coefficients=True)
    """
    # Convert to arrays
    lengths = np.array([length for length, _ in results])
    times = np.array([runtime for _, runtime in results])

    # Filter results where length is between MIN and MAX
    mask = (lengths >= MIN) & (lengths <= MAX)
    lengths = lengths[mask]
    times = times[mask]

    # Compute statistics by length
    unique_lengths = sorted(set(lengths)) 
    means = []
    ci_lows = []
    ci_highs = []
    counts = []

    from scipy import stats

    for length in unique_lengths:
        length_times = times[lengths == length]
        mean = np.mean(length_times)
        std = np.std(length_times, ddof=1)  # Sample std deviation
        n = len(length_times)

        # Compute 95% confidence interval
        if n > 1:
            # t-distribution critical value for 95% CI . The t-student distribution estimates the sample mean.
            t_critical = stats.t.ppf(0.975, df=n-1)  # 0.975 for two-tailed 95% CI
            margin = t_critical * std / np.sqrt(n) # Translate the 
            ci_low = mean - margin
            ci_high = mean + margin
        else: # If we have just one data point, we cannot compute the std.
            ci_low = mean
            ci_high = mean

        means.append(mean)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        counts.append(n)

    means = np.array(means)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    # Compute regression on log-log scale
    coefficients = None
    x_fit = None
    y_fit = None
    if show_regression:
        result = compute_loglog_regression(unique_lengths, means)
        if result[0] is not None:
            exponent_b, coefficient_a, r_squared, x_fit, y_fit = result
            coefficients = {
                'exponent': exponent_b,
                'coefficient': coefficient_a,
                'r_squared': r_squared,
                'equation': f'y = {coefficient_a:.4e} * x^{exponent_b:.4f}',
                'x_fit': x_fit,
                'y_fit': y_fit
            }
            print(f"Regression coefficients: {coefficients}")

    # Create or use existing axis
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Prefix Length (tokens)', fontsize=12)
        ax.set_ylabel('Runtime (ms)', fontsize=12)
        ax.set_title('Prefix Parsing Runtime (Log-Log)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        create_new = True
    else:
        create_new = False

    # # Plot individual points (optional, can be commented out if too cluttered)
    # if label:
    #     scatter_label = None  # Don't show individual points in legend to reduce clutter
    # else:
    #     scatter_label = 'Individual prefixes'
    # ax.scatter(lengths, times, alpha=scatter_alpha, s=1, color=color, label=scatter_label)

    # Plot mean line
    mean_label = f'{label} - Mean' if label else 'Mean'
    ax.plot(unique_lengths, means, 'o-',
             markersize=3, linewidth=2, alpha=0.9, color=color,
             label=mean_label, zorder=10)

    # Plot regression line if computed
    if show_regression and coefficients is not None:
        exponent_b = coefficients['exponent']
        coefficient_a = coefficients['coefficient']
        r_squared = coefficients['r_squared']
        x_fit = coefficients['x_fit']
        y_fit = coefficients['y_fit']
        # Include coefficient (a) in the label
        reg_label = f'{label} - Fit (a={coefficient_a:.2e}, b={exponent_b:.2f}, R²={r_squared:.3f})' if label else f'Fit (a={coefficient_a:.2e}, b={exponent_b:.2f}, R²={r_squared:.3f})'
        ax.plot(x_fit, y_fit, '--', color=color, linewidth=2.5, 
               alpha=0.8, label=reg_label, zorder=8)

    # Shaded area for 95% confidence interval
    ci_label = f'{label} - 95% CI' if label else '95% CI'
    ax.fill_between(unique_lengths, ci_lows, ci_highs,
                     alpha=0.2, color=color, label=ci_label, zorder=5)

    # If creating new plot, add reference lines and save
    if create_new:
        # Add reference lines
        x_range = np.array(unique_lengths)
        y_n = (means[0] / unique_lengths[0]) * x_range
        y_n2 = (means[0] / (unique_lengths[0]**2)) * (x_range**2)
        ax.plot(x_range, y_n, '--', color='gray', alpha=0.5, linewidth=1, label='O(n)', zorder=1)
        ax.plot(x_range, y_n2, '--', color='gray', alpha=0.5, linewidth=1, label='O(n²)', zorder=1)
        
        ax.legend(fontsize=15)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    if show_regression:
        return ax, coefficients
    else:
        return ax


def print_statistics(results):
    """Print summary statistics."""
    lengths = np.array([length for length, _ in results])
    times = np.array([runtime for _, runtime in results])

    unique_lengths = sorted(set(lengths))

    print(f"\n{'Length':<10} {'Count':<10} {'Mean (ms)':<15} {'Std (ms)':<15} {'Min (ms)':<15} {'Max (ms)'}")
    print("-" * 85)

    for length in unique_lengths:
        length_times = times[lengths == length]
        print(f"{length:<10} {len(length_times):<10} {np.mean(length_times):<15.6f} "
              f"{np.std(length_times):<15.6f} {np.min(length_times):<15.6f} {np.max(length_times):<15.6f}")

    print(f"\n{'='*85}")
    print(f"Total datapoints: {len(results)}")
    print(f"Overall mean: {np.mean(times):.6f} ms")
    print(f"Overall median: {np.median(times):.6f} ms")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean prefix parsing benchmark (parallelized)")
    parser.add_argument("--grammar", required=True)
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--output", default="benchmark_prefix_loglog.png")
    parser.add_argument("--n-workers", type=int, default=None,
                       help="Number of parallel workers (default: number of CPUs)")

    args = parser.parse_args()

    sentences = load_sentences(args.sentences, args.max_sentences, args.min_length)

    print(f"Sentences: {len(sentences)} (length >= {args.min_length})")
    
    n_workers = args.n_workers or multiprocessing.cpu_count()
    print(f"Running benchmarks in parallel with {n_workers} workers...\n")
    
    # Run prefix parsing benchmark
    print("Running prefix parsing benchmark...")
    start_time = time.time()
    results_prefix = benchmark_prefix_parsing_parallel(args.grammar, sentences, n_workers)
    prefix_time = time.time() - start_time
    print(f"Prefix parsing completed in {prefix_time:.2f} seconds")
    
    # Run regular parsing benchmark
    print("Running regular parsing benchmark...")
    start_time = time.time()
    results_regular = benchmark_parsing_parallel(args.grammar, sentences, n_workers)
    regular_time = time.time() - start_time
    print(f"Regular parsing completed in {regular_time:.2f} seconds")

    # Create single figure with both parsing results
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Length (terminals)', fontsize=24)
    ax.set_ylabel('Runtime (ms)', fontsize=24)
    ax.set_title('Parsing Runtime Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Plot prefix parsing results
    ax, coeffs_prefix = plot_results(results_prefix, ax=ax, label="Prefix Parsing", 
                                     color='blue', scatter_alpha=0.05)

    # Plot regular parsing results on same axis
    ax, coeffs_regular = plot_results(results_regular, ax=ax, label="Regular Parsing", 
                                     color='red', scatter_alpha=0.05)

    # Finalize and save combined figure
    ax.legend(fontsize=16, loc='best')
    plt.savefig(args.output, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"\nCombined plot saved to {args.output}")
    
    # Print regression results
    print("\n" + "="*70)
    print("Power Law Regression Results (y = a * x^b)")
    print("="*70)
    if coeffs_prefix:
        print(f"\nPrefix Parsing:")
        print(f"  Equation: {coeffs_prefix['equation']}")
        print(f"  Exponent (b): {coeffs_prefix['exponent']:.4f}")
        print(f"  Coefficient (a): {coeffs_prefix['coefficient']:.4e}")
        print(f"  R²: {coeffs_prefix['r_squared']:.4f}")
    if coeffs_regular:
        print(f"\nRegular Parsing:")
        print(f"  Equation: {coeffs_regular['equation']}")
        print(f"  Exponent (b): {coeffs_regular['exponent']:.4f}")
        print(f"  Coefficient (a): {coeffs_regular['coefficient']:.4e}")
        print(f"  R²: {coeffs_regular['r_squared']:.4f}")
    
    print("\nPrefix Parsing Statistics:")
    print_statistics(results_prefix)
    print("\n" + "="*50)
    print("Regular Parsing Statistics:")
    print_statistics(results_regular)


if __name__ == "__main__":
    main()
