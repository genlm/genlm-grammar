#!/usr/bin/env python3
"""
Empirical grammar blowup: size and nonterminal count through four
preprocessing pipelines, in the order the actual parsers use.

  1. CKY(G)           : .cnf
  2. CKY(PG(G))       : prefix_grammar, .cnf
  3. Earley(G)        : nullaryremove(binarize=True), unarycycleremove,
                        renumber, trim  (matches parse/earley.py:72)
  4. Earley(PG(G))    : prefix_grammar, then the same Earley chain

The code applies PG to the raw grammar (see benchmarks/benchmark.py:69);
binarization happens later, inside nullaryremove.

Usage:
    python benchmarks/benchmark_grammar_size.py \\
        --grammar earleyx/grammars/wsj500unk.grammar --start ROOT
"""
import argparse, json, signal, time
from contextlib import contextmanager
from pathlib import Path

from genlm.grammar.treebank import TreebankCFG


@contextmanager
def alarm(seconds):
    def _h(_s, _f): raise TimeoutError(f"exceeded {seconds}s")
    prev = signal.signal(signal.SIGALRM, _h)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def stats(g):
    return {"size": g.size, "rules": g.num_rules,
            "N": len(g.N), "V": len(g.V)}


def fmt(s):
    return (f"|G|={s['size']:>11,}  |R|={s['rules']:>10,}  "
            f"|N|={s['N']:>7,}  |V|={s['V']:>7,}")


CKY = [
    ("separate_terminals",           lambda g: g.separate_terminals()),
    ("nullaryremove(binarize=True)", lambda g: g.nullaryremove(binarize=False)),
    ("trim",                         lambda g: g.trim()),
    ("unaryremove",                  lambda g: g.unaryremove()),
    ("trim",                         lambda g: g.trim()),
]
EARLEY = [
    ("nullaryremove(binarize=True)", lambda g: g.nullaryremove(binarize=True)),
    ("unarycycleremove",             lambda g: g.unarycycleremove()),
    ("renumber",                     lambda g: g.renumber()),
    ("trim",                         lambda g: g.trim()),
]
PG_PRELUDE = [
    ("prefix_grammar",               lambda g: g.prefix_grammar),
]
BINARIZE = [
    ("binarize",  lambda g: g.binarize())
]
PIPELINES = [
    ("CKY(G)",          BINARIZE + CKY),
    ("CKY(PG(G))",      BINARIZE + PG_PRELUDE + CKY),
    ("Earley(G)",       EARLEY),
    ("Earley(PG(G))",   PG_PRELUDE + EARLEY),
]


def run(label, g, stages, limit):
    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    log = [{"stage": "input", "stats": stats(g), "t": 0.0}]
    print(f"  {'input':40s} {fmt(log[0]['stats'])}")
    for name, fn in stages:
        t0 = time.time()
        try:
            with alarm(limit):
                g = fn(g)
        except TimeoutError:
            dt = time.time() - t0
            print(f"  {name:40s} TIMEOUT after {dt:.1f}s")
            log.append({"stage": name, "status": "timeout", "t": dt})
            return log, False
        except Exception as e:
            dt = time.time() - t0
            print(f"  {name:40s} ERROR after {dt:.1f}s: {type(e).__name__}: {e}")
            log.append({"stage": name, "status": "error",
                        "error": repr(e), "t": dt})
            return log, False
        dt = time.time() - t0
        log.append({"stage": name, "stats": stats(g), "t": dt})
        print(f"  {name:40s} {fmt(log[-1]['stats'])}  ({dt:.2f}s)")
    return log, True


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grammar", required=True)
    p.add_argument("--start", default="ROOT")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-stage timeout (s), default 1800")
    p.add_argument("--json-out", default=None)
    p.add_argument("--only", nargs="+", default=None,
                   choices=[label for label, _ in PIPELINES],
                   help="only run the specified pipelines")
    args = p.parse_args()
    pipelines = ([p for p in PIPELINES if p[0] in args.only]
                 if args.only else PIPELINES)

    print(f"Grammar           : {args.grammar}")
    print(f"Start             : {args.start}")
    print(f"Per-stage timeout : {args.timeout}s")
    text = Path(args.grammar).read_text()

    results = []
    for label, stages in pipelines:
        g = TreebankCFG.from_string(text, start=args.start)
        log, ok = run(f"[{Path(args.grammar).name}] {label}",
                      g, stages, args.timeout)
        results.append({"label": label, "log": log, "ok": ok})

    print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
    for r in results:
        last = r["log"][-1]
        if r["ok"]:
            print(f"  {r['label']:22s} {fmt(last['stats'])}")
        else:
            print(f"  {r['label']:22s} failed at {last['stage']}")

    by_label = {r["label"]: r for r in results}
    print("\nBlowup ratios (PG / plain):")
    for plain, pg, name in [("CKY(G)", "CKY(PG(G))", "CKY"),
                            ("Earley(G)", "Earley(PG(G))", "Earley")]:
        a, b = by_label.get(plain), by_label.get(pg)
        if a and b and a["ok"] and b["ok"]:
            sa, sb = a["log"][-1]["stats"], b["log"][-1]["stats"]
            print(f"  {name:8s}  size = {sb['size']/sa['size']:.3f}x  "
                  f"|N| = {sb['N']/sa['N']:.3f}x")
        elif a is None or b is None:
            continue
        else:
            print(f"  {name:8s}  incomplete")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"grammar": args.grammar, "start": args.start,
             "timeout_s": args.timeout, "results": results}, indent=2))
        print(f"\nWrote: {args.json_out}")


if __name__ == "__main__":
    main()
