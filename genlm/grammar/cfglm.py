"""
Fast computation of the posterior distrubtion over the next word in a WCFG language model.
"""

from genlm.grammar.cfg import CFG, _gen_nt
from genlm.grammar.lm import LM
from genlm.grammar.semiring import Boolean, Float


EOS = "▪"


def locally_normalize(self, **kwargs):
    """Locally normalize the grammar's rule weights.

    Returns a transformed grammar where:
    1. The total weight of rules with the same head symbol sums to one
    2. Each derivation's weight is proportional to the original grammar
       (differs only by a multiplicative normalization constant)

    Args:
        **kwargs: Additional arguments passed to self.agenda()

    Returns:
        (CFG): A new grammar with locally normalized weights
    """
    new = self.spawn()
    Z = self.agenda(**kwargs)
    for r in self:
        if Z[r.head] == 0:
            continue
        new.add(r.w * Z.product(r.body) / Z[r.head], r.head, *r.body)
    return new


def add_EOS(cfg, eos=None):
    """Add an end-of-sequence symbol to a CFG's language.

    Transforms the grammar to append the EOS symbol to every string it generates.

    Args:
        cfg (CFG): The input grammar
        eos (optional): The end-of-sequence symbol to add. Defaults to ▪.

    Returns:
        (CFG): A new grammar that generates strings ending in EOS

    Raises:
        AssertionError: If EOS is already in the grammar's vocabulary

    """
    S = _gen_nt("<START>")
    new = cfg.spawn(S=S)
    eos = eos or EOS
    assert eos not in cfg.V
    new.V.add(eos)
    new.add(cfg.R.one, S, cfg.S, eos)
    for r in cfg:
        new.add(r.w, r.head, *r.body)
    return new


class BoolCFGLM(LM):
    """Language model interface for Boolean-weighted CFGs.

    Uses Earley's algorithm or CKY for inference. The grammar is converted to use
    Boolean weights if needed, where positive weights become True and zero/negative
    weights become False.

    Args:
        cfg (CFG): The context-free grammar to use
        alg (str): Parsing algorithm to use - either 'earley' or 'cky'

    Raises:
        ValueError: If alg is not 'earley' or 'cky'
    """

    def __init__(self, cfg, alg="earley"):
        """Initialize a BoolCFGLM.

        Args:
            cfg (CFG): The context-free grammar to use as the language model
            alg (str): Parsing algorithm to use - either 'earley' or 'cky'

        Raises:
            ValueError: If alg is not 'earley' or 'cky'
        """
        if EOS not in cfg.V:
            cfg = add_EOS(cfg, eos=EOS)
        if cfg.R != Boolean:
            cfg = cfg.map_values(lambda x: Boolean(x > 0), Boolean)
        if alg == "earley":
            from genlm.grammar.parse.earley import Earley

            self.model = Earley(cfg.prefix_grammar)
        elif alg == "cky":
            from genlm.grammar.parse.cky import CKYLM

            self.model = CKYLM(cfg)
        else:
            raise ValueError(f"unrecognized option {alg}")
        super().__init__(eos=EOS, V=cfg.V)

    def p_next(self, context):
        """Compute next token probabilities given a context.

        Args:
            context (sequence): The conditioning context

        Returns:
            (Float.chart): The next token weights

        Raises:
            AssertionError: If context contains out-of-vocabulary tokens
        """
        assert set(context) <= self.V, f"OOVs detected: {set(context) - self.V}"
        p = self.model.next_token_weights(self.model.chart(context)).trim()
        return Float.chart({w: 1 for w in p})

    def __call__(self, context):
        """Check if a context is possible under this grammar.

        Args:
            context (sequence): The context to check

        Returns:
            (bool): True if the context has non-zero weight
        """
        return float(super().__call__(context) > 0)

    def clear_cache(self):
        """Clear any cached computations."""
        self.model.clear_cache()

    @classmethod
    def from_string(cls, x, semiring=Boolean, **kwargs):
        """Create a BoolCFGLM from a string representation of a grammar.

        Args:
            x (str): The grammar string
            semiring: The semiring for weights (default: Boolean)
            **kwargs: Additional arguments passed to __init__

        Returns:
            (BoolCFGLM): A new language model
        """
        return cls(CFG.from_string(x, semiring), **kwargs)
