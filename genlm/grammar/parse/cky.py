from collections import defaultdict

from genlm.grammar.lm import LM
from genlm.grammar.cfglm import EOS, add_EOS, locally_normalize, CFG
from genlm.grammar import Float


class CKYLM(LM):
    """Probabilistic Context-Free Grammar Language Model.

    Uses CKY parsing algorithm and prefix grammar transformation for efficient inference
    over context-free grammars.

    Args:
        cfg (CFG): The context-free grammar to use as the language model
        **kwargs: Additional arguments passed to IncrementalCKY

    Attributes:
        cfg (CFG): The original context-free grammar
        pfg (CFG): The prefix grammar in CNF form used for incremental parsing
        model (IncrementalCKY): The incremental CKY parser for computing probabilities
    """

    def __init__(self, cfg, **kwargs):
        """Initialize a CKY-based language model.

        Args:
            cfg (CFG): The context-free grammar to use as the language model. Will be
                converted to CNF and prefix form for incremental parsing.
            **kwargs: Additional arguments passed to IncrementalCKY parser initialization

        Raises:
            AssertionError: If EOS token not in grammar vocabulary
        """
        if EOS not in cfg.V:
            cfg = add_EOS(cfg)
        self.cfg = cfg
        self.pfg = self.cfg.cnf.prefix_grammar.cnf
        self.model = IncrementalCKY(self.pfg, **kwargs)
        super().__init__(V=cfg.V, eos=EOS)

    def p_next(self, context):
        """Compute probability distribution over next tokens given a context.

        Args:
            context: Sequence of tokens representing the prefix

        Returns:
            Normalized probability distribution over possible next tokens

        Raises:
            AssertionError: If context contains tokens not in vocabulary
        """
        assert set(context) <= self.V, f"OOVs detected: {set(context) - self.V}"
        return self.model.p_next(context).normalize()

    @classmethod
    def from_string(cls, x, semiring=Float, **kwargs):
        """Create a CKYLM from a grammar string representation.

        Args:
            x (str): String representation of the grammar
            semiring: Semiring to use for weights (default: Float)
            **kwargs: Additional arguments for grammar normalization

        Returns:
            CKYLM: A new language model instance
        """
        return cls(locally_normalize(CFG.from_string(x, semiring), **kwargs))

    def clear_cache(self):
        """Clear the parser's chart cache."""
        self.model.clear_cache()


class IncrementalCKY:
    """
    An incremental CKY parser implementation for Context-Free Grammars (CFG).

    This parser maintains a chart cache for efficient incremental parsing and supports
    weight computations for next-token predictions.
    """

    def __init__(self, cfg):
        """
        Initialize an incremental CKY parser.

        Args:
            cfg (CFG): The context-free grammar
        """
        cfg = cfg.renumber()
        self.cfg = cfg
        self.S = cfg.S

        # cache columns of the chart indexed by prefix
        self._chart = {}

        [self.nullary, self.terminal, binary] = cfg._cnf
        r_y_xz = defaultdict(list)
        for r in binary:  # binary rules
            r_y_xz[r.body[0]].append(r)
        self.r_y_xz = r_y_xz

    def clear_cache(self):
        self._chart.clear()

    def __call__(self, x):
        return self.chart(x)[len(x)][0][self.S]

    def p_next(self, prefix):
        """
        Compute the weights for all possible next tokens given a prefix.

        Args:
            prefix: The current sequence of tokens

        Returns:
            Dictionary mapping possible next tokens to their weights
        """
        return self.next_token_weights(self.chart(prefix), prefix)

    def chart(self, prefix):
        """
        Get the parsing chart for a given prefix, computing it if not cached.

        Args:
            prefix: The sequence of tokens to parse

        Returns:
            The CKY chart for the prefix
        """
        c = self._chart.get(prefix)
        if c is None:
            c = self._compute_chart(prefix)
            self._chart[prefix] = c
        return c

    def _compute_chart(self, prefix):
        """
        Compute the CKY chart for a given prefix.

        Args:
            prefix: The sequence of tokens to parse

        Returns:
            A new chart for the prefix, either initialized for empty prefix
            or extended from the previous chart
        """
        if len(prefix) == 0:
            tmp = [defaultdict(self.cfg.R.chart)]
            tmp[0][0][self.cfg.S] = self.nullary
            return tmp
        else:
            chart = self.chart(prefix[:-1])
            last_chart = self.extend_chart(chart, prefix)
            return chart + [
                last_chart
            ]  # TODO: avoid list addition here as it is not constant time!

    def next_token_weights(self, chart, prefix):
        """
        Compute the total weight for each possible next token following the prefix.

        An O(N²) time algorithm that calculates the total weight of a each next-token
        extension of `prefix`.

        Args:
            chart: The current CKY chart
            prefix: The current sequence of tokens

        Returns:
            (Chart) Dictionary mapping possible next tokens to their weights # XXX
        """
        k = len(prefix) + 1

        cfg = self.cfg
        terminal = self.terminal
        r_y_xz = self.r_y_xz

        # the code below is just backprop / outside algorithm
        α = defaultdict(cfg.R.chart)
        α[0][cfg.S] += cfg.R.one

        # Binary rules
        for span in reversed(range(2, k + 1)):
            i = k - span
            α_i = α[i]
            for j in range(i + 1, k):
                chart_ij = chart[j][i]

                α_j = α[j]
                for Y, y in chart_ij.items():
                    for r in r_y_xz[Y]:
                        X = r.head
                        Z = r.body[1]
                        α_j[Z] += r.w * y * α_i[X]

        # Preterminal
        q = cfg.R.chart()
        tmp = α[k - 1]
        for w in cfg.V:
            for r in terminal[w]:
                q[w] += r.w * tmp[r.head]

        return q

    def extend_chart(self, chart, prefix):
        """
        An O(N²) time algorithm that extends the parsing chart with the last token of the prefix.

        Args:
            chart: The current CKY chart
            prefix: The sequence of tokens including the new token

        Returns:
            A new chart column incorporating the last token
        """
        k = len(prefix)

        cfg = self.cfg
        r_y_xz = self.r_y_xz

        new = defaultdict(cfg.R.chart)

        # Nullary
        new[k][cfg.S] += self.nullary

        # Preterminal
        tmp = new[k - 1]
        for r in self.terminal[prefix[k - 1]]:
            tmp[r.head] += r.w

        # Binary rules
        for span in range(2, k + 1):
            i = k - span
            new_i = new[i]
            for j in range(i + 1, k):
                chart_ij = chart[j][i]
                new_j = new[j]
                for Y, y in chart_ij.items():
                    for r in r_y_xz[Y]:
                        X = r.head
                        Z = r.body[1]
                        z = new_j[Z]
                        x = r.w * y * z
                        new_i[X] += x

        return new
