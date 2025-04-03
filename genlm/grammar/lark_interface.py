import lark
import string
import interegular
from interegular.fsm import anything_else

import arsenal
import warnings
from collections import Counter

from genlm.grammar import WFSA, Float
from genlm.grammar.cfg import CFG, Rule


class LarkStuff:
    """Utility class for leveraging lark as a front-end syntax for specifying grammars.

    This class provides functionality to convert Lark grammars into genlm_grammar format,
    handling various features and edge cases in the conversion process.

    Attributes:
        raw_grammar: The original Lark grammar string
        terminals: Terminal symbols from the Lark grammar
        ignore_terms: Terms marked with 'ignore' directive in Lark
        rules: Grammar production rules

    Warning:
        The tokenization semantics may differ from Lark since prioritized/maximum-munch
        tokenization is not preserved when encoding into the grammar.

    Note:
        Several features require careful handling in the conversion:

        - The 'ignore' directive is implemented by concatenating terminal regexes with
          an optional prefix containing ignore terms. This preserves semantics but uses
          a different implementation.

        - When compiling terminal regexes to Python re syntax, some features may not be
          fully supported by the interegular library.

        - Implementations of '.' and '^' use negated character classes relative to
          string.printable. Other edge cases may exist, particularly with lookahead/lookbehind.
    """

    __slots__ = (
        "raw_grammar",
        "terminals",
        "ignore_terms",
        "rules",
    )

    def __init__(self, grammar, cnf=False):
        """Initialize a LarkStuff instance.

        Args:
            grammar: A Lark grammar string
            cnf: Whether to convert grammar to Chomsky Normal Form

        Raises:
            ValueError: If grammar does not define a 'start' rule
        """
        self.raw_grammar = grammar

        builder = lark.load_grammar.GrammarBuilder()
        builder.load_grammar(grammar)
        lark_grammar = builder.build()

        if not any(
            rule.value == "start"
            for rule in lark_grammar.rule_defs[0]
            if isinstance(rule, lark.lexer.Token)
        ):
            raise ValueError("Grammar must define a `start` rule")

        terminals, rules, ignores = lark_grammar.compile(["start"], set())

        if cnf:
            parser = lark.parsers.cyk.Parser(rules)
            self.rules = parser.grammar.rules
        else:
            self.rules = rules

        self.terminals = terminals
        self.ignore_terms = ignores

    def convert(self):
        """Convert the Lark grammar into a genlm_grammar.CFG grammar.

        Returns:
            CFG: A context-free grammar in genlm_grammar format with renumbered states
        """
        try:
            rules = [
                Rule(1, r.lhs.name, tuple(y.name for y in r.rhs)) for r in self.rules
            ]
        except AttributeError:
            rules = [
                Rule(1, r.origin.name, tuple(y.name for y in r.expansion))
                for r in self.rules
            ]

        lhs_count = Counter([r.head for r in rules])
        cfg = CFG(R=Float, S="start", V={t.name for t in self.terminals})
        for r in rules:
            cfg.add(1 / lhs_count[r.head], r.head, *r.body)
        return cfg.renumber()

    def char_cfg(self, *args, **kwargs):
        return self._char_cfg(*args, **kwargs, to_bytes=False)

    def byte_cfg(self, *args, **kwargs):
        return self._char_cfg(*args, **kwargs, to_bytes=True)

    def _char_cfg(
        self, decay=1, delimiter="", charset="core", recursion="right", to_bytes=False
    ):
        """Convert to a character- or byte-level CFG with optional ignore patterns.

        Args:
            decay: Weight decay factor for rules
            delimiter: Delimiter between tokens (not currently supported)
            charset: Character set to use ('core' or custom set)
            recursion: Direction of recursion ('right' or 'left')
            to_bytes: Whether to convert to a byte-level CFG

        Returns:
            CFG: A character- or byte-level context-free grammar

        Raises:
            NotImplementedError: If delimiter is non-empty
        """
        if delimiter != "":
            raise NotImplementedError(f"{delimiter = !r} is not supported.")

        cfg = self.convert()

        # rename all of the internals to avoid naming conflicts.
        _f = arsenal.Integerizer()

        def f(x):
            return f"N{_f(x)}"

        foo = CFG(Float, S=f(cfg.S), V=set())
        for r in cfg:
            foo.add(r.w * decay, f(r.head), *(f(y) for y in r.body))
        del r

        if self.ignore_terms:
            # union of ignore patterns
            IGNORE = "$IGNORE"
            assert IGNORE not in cfg.V
            ignore = f(IGNORE)
            foo.add(decay, ignore)
            for token_class in self.terminals:
                if token_class.name not in self.ignore_terms:
                    continue
                foo.add(decay, ignore, f(token_class.name))

        for token_class in self.terminals:
            regex = token_class.pattern.to_regexp()

            fsa = interegular_to_wfsa(
                regex,
                name=lambda x, t=token_class.name: f((t, x)),
                charset=charset,
            )

            if to_bytes:
                fsa = fsa.to_bytes()

            if token_class.name in self.ignore_terms or not self.ignore_terms:
                G = fsa.to_cfg(S=f(token_class.name), recursion=recursion)

                foo.V |= G.V
                for r in G:
                    foo.add(r.w * decay, r.head, *r.body)

            else:
                tmp = f(("tmp", token_class.name))
                G = fsa.to_cfg(S=tmp, recursion=recursion)

                foo.V |= G.V
                for r in G:
                    foo.add(r.w * decay, r.head, *r.body)

                foo.add(decay, f(token_class.name), ignore, tmp)

        assert len(foo.N & foo.V) == 0

        return foo


def interegular_to_wfsa(pattern, charset="core", name=lambda x: x):
    """Convert an interegular regex pattern to a weighted finite state automaton.

    Args:
        pattern: The regex pattern string to convert
        name: Function to transform state names (default: identity function)
        charset: Character set to use for negative character classes. Can be 'core' for `string.printable`,
                or a custom set of characters. This is the set of characters against which negative character
                classes are matched.

    Returns:
        (WFSA): A weighted finite state automaton representing the regex pattern

    Raises:
        NotImplementedError: If charset is not 'core' or a set

    Note:
        Multi-character transitions from the regex are excluded with a warning, as they
        cannot be directly represented in the WFSA format.
    """
    if charset == "core":
        charset = set(string.printable)
    elif isinstance(charset, set):
        pass
    else:
        # TODO: implement other charsets
        raise NotImplementedError(f"charset {charset} not implemented")

    # Compile the regex pattern to an FSM
    fsm = interegular.parse_pattern(pattern).to_fsm()

    def expand_alphabet(a):
        if anything_else in fsm.alphabet.by_transition[a]:
            assert fsm.alphabet.by_transition[a] == [anything_else]
            return charset - set(fsm.alphabet)
        else:
            return fsm.alphabet.by_transition[a]

    if 0:
        from fsa import FSA

        m = FSA()
        m.add_start(name(fsm.initial))

        rejection_states = [e for e in fsm.states if not fsm.islive(e)]
        for i in fsm.states:
            if i in fsm.finals:
                m.add_stop(name(i))
            for a, j in fsm.map[i].items():
                if j in rejection_states:
                    continue
                for A in expand_alphabet(a):
                    if len(A) != 1:
                        warnings.warn(
                            f"Excluding multi-character arc {A!r} in pattern {pattern!r} (possibly a result of case insensitivity of arcs {expand_alphabet(a)})"
                        )
                    m.add(name(i), A, name(j))

        # DFA minimization
        M = m.min()

        del m
        del fsm

        m = WFSA(Float)
        for i in M.nodes:
            K = len(list(M.arcs(i))) + (i in M.stop)
            if i in M.start:
                m.add_I(name(i), 1)
            if i in M.stop:
                m.add_F(name(i), 1 / K)
            for a, j in M.arcs(i):
                m.add_arc(name(i), a, name(j), 1 / K)
        return m

    else:
        m = WFSA(Float)
        m.add_I(name(fsm.initial), 1)

        rejection_states = [e for e in fsm.states if not fsm.islive(e)]
        for i in fsm.states:
            # determine this state's fan out
            K = 0
            for a, j in fsm.map[i].items():
                # print(f'{i} --{a}/{fsm.alphabet.by_transition[a]}--> {j}')
                if j in rejection_states:
                    continue
                for A in expand_alphabet(a):
                    assert isinstance(A, str)
                    if len(A) != 1:
                        warnings.warn(
                            f"Excluding multi-character arc {A!r} in pattern {pattern!r} (possibly a result of case insensitivity of arcs {expand_alphabet(a)})"
                        )
                        continue
                    K += 1
            if i in fsm.finals:
                K += 1
            if K == 0:
                continue
            if i in fsm.finals:
                m.add_F(name(i), 1 / K)
            for a, j in fsm.map[i].items():
                if j in rejection_states:
                    continue
                for A in expand_alphabet(a):
                    m.add_arc(name(i), A, name(j), 1 / K)

        return m
