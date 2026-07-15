"""
Thin wrapper that exposes the Rust rescaled Earley parser with the same
interface as the Python rescaled Earley class. Grammar preprocessing stays
in Python; the hot path runs in Rust.
"""

import numpy as np
from arsenal import Integerizer
from collections import defaultdict

from genlm.grammar.cfglm import EOS, add_EOS, locally_normalize
from genlm.grammar.lm import LM
from genlm.grammar.semiring import Float
from genlm.grammar.cfg import CFG
from genlm.grammar.linear import WeightedGraph
from genlm.grammar.semiring import Boolean
from genlm.grammar.util import DEFAULT_MAX_CACHE_SIZE

try:
    from genlm_earley import RustEarleyRescaled
except ImportError:
    RustEarleyRescaled = None

_sum = lambda x, R: x[0] + _sum(x[1:], R) if len(x) > 0 else R.zero


class EarleyRescaledRust:
    """
    Drop-in replacement for the Python rescaled Earley class that delegates
    the hot path to a Rust implementation via PyO3.

    Note: preprocessing does NOT call .trim() (matching the Python rescaled version).

    `max_cache_size` bounds the number of cached prefixes (LRU eviction);
    defaults to 10,000, None means unbounded. Eviction (like clear_cache)
    invalidates chart handles returned by earlier `chart()` calls; using an
    expired handle raises ValueError rather than returning wrong results.
    """

    def __init__(self, cfg, max_cache_size=DEFAULT_MAX_CACHE_SIZE):
        if RustEarleyRescaled is None:
            raise ImportError(
                "genlm_earley Rust extension not found. "
                "Build with: cd rust && maturin develop --release"
            )
        if max_cache_size is not None and max_cache_size < 1:
            raise ValueError("max_cache_size must be ≥ 1 or None (unbounded)")

        # Note: no .trim() — matches Python rescaled Earley
        cfg = cfg.nullaryremove(binarize=True).unarycycleremove().renumber()
        self.cfg = cfg

        # ── Same preprocessing as Python Earley ────────────────────────────
        order = cfg._unary_graph_transpose().buckets
        ORDER_MAX = max(order.values())

        R_outgoing = defaultdict(set)
        for r in cfg:
            if len(r.body) == 0:
                continue
            A = r.head
            B = r.body[0]
            if cfg.is_terminal(B):
                continue
            R_outgoing[A].add(B)

        intern_Ys = Integerizer()
        assert intern_Ys(()) == 0

        for r in cfg:
            for p in range(len(r.body) + 1):
                intern_Ys.add(r.body[p:])

        rhs = {}
        for X in cfg.N:
            rhs[X] = []
            for r in cfg.rhs[X]:
                if r.body == ():
                    continue
                rhs[X].append((float(r.w), int(intern_Ys(r.body))))

        first_Ys_raw = [None] * len(intern_Ys)
        rest_Ys = [0] * len(intern_Ys)
        unit_Ys = [False] * len(intern_Ys)

        for Ys, code in list(intern_Ys.items()):
            unit_Ys[code] = len(Ys) == 1
            if len(Ys) > 0:
                first_Ys_raw[code] = Ys[0]
                rest_Ys[code] = int(intern_Ys(Ys[1:]))

        # ── Build unified u32 symbol encoding ──────────────────────────────
        max_nt = max(cfg.N) if cfg.N else 0
        terminal_to_id = {}
        id_to_terminal = {}
        for i, t in enumerate(sorted(cfg.V)):
            tid = max_nt + 1 + i
            terminal_to_id[t] = tid
            id_to_terminal[tid] = t

        first_ys_u32 = []
        is_terminal_flags = []
        for code in range(len(intern_Ys)):
            sym = first_Ys_raw[code]
            if sym is None:
                first_ys_u32.append(0)
                is_terminal_flags.append(False)
            elif isinstance(sym, str):
                first_ys_u32.append(terminal_to_id[sym])
                is_terminal_flags.append(True)
            else:
                first_ys_u32.append(int(sym))
                is_terminal_flags.append(False)

        outgoing = {int(k): [int(v) for v in vs] for k, vs in R_outgoing.items()}
        nonterminals = set(int(n) for n in cfg.N)

        empty_weight = float(
            _sum([r.w for r in cfg.rhs[cfg.S] if r.body == ()], cfg.R)
        )

        # ── Construct the Rust engine ──────────────────────────────────────
        self._rust = RustEarleyRescaled(
            rhs={int(k): v for k, v in rhs.items()},
            start=int(cfg.S),
            order={int(k): int(v) for k, v in order.items()},
            order_max=int(ORDER_MAX),
            outgoing=outgoing,
            first_ys=first_ys_u32,
            is_terminal_flags=is_terminal_flags,
            rest_ys=rest_Ys,
            unit_ys=unit_Ys,
            terminal_to_id=terminal_to_id,
            id_to_terminal=id_to_terminal,
            nonterminals=nonterminals,
            empty_weight=empty_weight,
            max_cache_size=max_cache_size,
        )

        self._terminal_to_id = terminal_to_id
        self.V = cfg.V

    def __call__(self, x):
        """Parse token sequence x, return its weight (unrescaled)."""
        tokens = [str(t) for t in x]
        return self._rust.parse(tokens)

    def logp(self, x):
        """Parse token sequence x, return log of its weight."""
        tokens = [str(t) for t in x]
        return self._rust.logp(tokens)

    def chart(self, x):
        """Compute chart for token sequence. Returns opaque chart reference."""
        tokens = [str(t) for t in x]
        return self._rust.chart(tokens)

    def next_token_weights(self, col_indices):
        """Compute next-token weights from a chart. Returns a normalized Chart."""
        raw = self._rust.next_token_weights(col_indices)
        result = self.cfg.R.chart()
        for terminal_str, weight in raw.items():
            result[terminal_str] = self.cfg.R(weight)
        return result

    def clear_cache(self):
        self._rust.clear_cache()

    @property
    def _chart(self):
        """Compatibility shim: benchmarks call _chart.clear()"""
        class _CacheProxy:
            def __init__(self, rust_obj):
                self._rust = rust_obj
            def clear(self):
                self._rust.clear_cache()
        return _CacheProxy(self._rust)


class EarleyRescaledRustLM(LM):
    """Language model using the Rust rescaled Earley backend."""

    def __init__(self, cfg, max_cache_size=DEFAULT_MAX_CACHE_SIZE):
        if EOS not in cfg.V:
            cfg = add_EOS(cfg)
        self.cfg = cfg
        self.model = EarleyRescaledRust(cfg.prefix_grammar, max_cache_size=max_cache_size)
        super().__init__(V=cfg.V, eos=EOS)

    def p_next(self, context):
        assert set(context) <= self.V, f"OOVs detected: {set(context) - self.V}"
        return self.model.next_token_weights(self.model.chart(context))

    def clear_cache(self):
        self.model.clear_cache()

    @classmethod
    def from_string(cls, x, semiring=Float, **kwargs):
        return cls(locally_normalize(CFG.from_string(x, semiring), **kwargs))
