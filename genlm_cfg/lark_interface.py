import lark
import string
import interegular
from interegular.fsm import anything_else

import arsenal
import warnings
from collections import Counter

from genlm_cfg import WFSA, Float
from genlm_cfg.cfg import CFG, Rule


class LarkStuff:
    """Utility class for leveraging the lark as a front-end syntax for specifying
    grammars.

    Warning: 
        There may be infelicity in the tokenization semantics as there is
        no longer a prioritized or maximum-munch semantics to the tokenizer when we
        encode it into the grammar.

    Note: 
        In conversion from lark to genlm_cfg, there are numerous features that
        need to be handled with care.

        * Notably, the `ignore` directive in lark is supported by concatenating
        existing terminal class regexes with an optional prefix containing the
        ignore terms. The semantics of this are equivalent, but the implementation
        is not.

        * When lark compiles terminal class regexes to python re syntax, not all
        features are supported by greenery.

        - Our implementations of `.` and `^` are in terms of negated character
            classes, and require special handling.  In our conversion, we consider
            negation with respect to a superset defined by `string.printable`. There
            may be other cases we have not yet encountered, so it is important to
            verify that conversions are correct when incorporating new grammars. We
            expect edge cases with lookahead and lookbehind assertions to be
            particularly problematic.

    TODO: update now that greenery has been replaced by interegular

    """

    __slots__ = (
        'raw_grammar',
        'terminals',
        'ignore_terms',
        'rules',
    )

    def __init__(self, grammar, cnf=False):
        self.raw_grammar = grammar

        builder = lark.load_grammar.GrammarBuilder()
        builder.load_grammar(grammar)
        lark_grammar = builder.build()

        if not any(
            rule.value == 'start'
            for rule in lark_grammar.rule_defs[0]
            if isinstance(rule, lark.lexer.Token)
        ):
            raise ValueError('Grammar must define a `start` rule')

        terminals, rules, ignores = lark_grammar.compile(['start'], set())

        if cnf:
            parser = lark.parsers.cyk.Parser(rules)
            # self.instance = lark.Lark(grammar, lexer='basic', parser='cyk')
            # self.lex = self.instance.lex
            self.rules = parser.grammar.rules

        else:
            # self.parser = lark.parsers.earley.Parser(rules)
            # self.instance = lark.Lark(grammar, parser='earley')
            # self.lex = self.instance.lex
            self.rules = rules

        self.terminals = terminals
        self.ignore_terms = ignores

    def convert(self):
        "Convert the lark grammar into a `genlm_cfg.CFG` grammar."

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
        cfg = CFG(R=Float, S='start', V={t.name for t in self.terminals})
        for r in rules:
            cfg.add(1 / lhs_count[r.head], r.head, *r.body)
        return cfg.renumber()

    def char_cfg(self, decay=1, delimiter='', charset='core', recursion='right'):
        if delimiter != '':
            raise NotImplementedError(f'{delimiter = !r} is not supported.')

        cfg = self.convert()

        # rename all of the internals to avoid naming conflicts.
        f = arsenal.Integerizer()

        foo = CFG(Float, S=f(cfg.S), V=set())
        for r in cfg:
            foo.add(r.w * decay, f(r.head), *(f(y) for y in r.body))
        del r

        if self.ignore_terms:
            # union of ignore patterns
            IGNORE = '$IGNORE'
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

            if token_class.name in self.ignore_terms or not self.ignore_terms:
                G = fsa.to_cfg(S=f(token_class.name), recursion=recursion)

                foo.V |= G.V
                for r in G:
                    foo.add(r.w * decay, r.head, *r.body)

            else:
                tmp = f(('tmp', token_class.name))
                G = fsa.to_cfg(S=tmp, recursion=recursion)

                foo.V |= G.V
                for r in G:
                    foo.add(r.w * decay, r.head, *r.body)

                foo.add(decay, f(token_class.name), ignore, tmp)

        assert len(foo.N & foo.V) == 0

        # assert len(foo) == len(foo.trim())

        #        if self.ignore_terms:
        #            old = self.char_cfg_old(decay=decay, delimiter=delimiter, charset=charset, recursion=recursion)
        #            print('old -> new:')
        #            print('  rules:', len(old), '->', len(foo))
        #            print('  size: ', old.size, '->', foo.size)
        #            print('  nts:  ', len(old.N), '->', len(foo.N))
        #            print('n terminal categories:', len(self.terminals))

        return foo


#    def char_cfg_old(self, decay=1, delimiter='', charset='core', recursion='right'):
#        if delimiter:
#            warnings.warn(
#                'Use of delimiter enforced between terminals. If delimiter is not a strict subset of `%ignore`, generated strings will deviate from original grammar.'
#            )
#
#        ignore_regex = f'(?:{"|".join([t.pattern.to_regexp() for t in self.terminals if t.name in self.ignore_terms])})?'
#
#        cfg = self.convert()
#
#        # rename all of the internals to avoid naming conflicts.
#        f = arsenal.Integerizer()
#
#        foo = CFG(Float, S=f(cfg.S), V=set())
#        for r in cfg:
#            foo.add(r.w * decay, f(r.head), *(f(y) for y in r.body))
#
#        for token_class in self.terminals:
#            if token_class.name in self.ignore_terms:
#                continue
#            regex = ignore_regex + token_class.pattern.to_regexp() + delimiter
#
#            fsa = interegular_to_wfsa(
#                regex,
#                name=lambda x, t=token_class.name: f((t, x)),
#                charset=charset,
#            )
#            G = fsa.to_cfg(S=f(token_class.name), recursion=recursion)
#
#            foo.V |= G.V
#            for r in G:
#                foo.add(r.w * decay, r.head, *r.body)
#
#        assert len(foo.N & foo.V) == 0
#
#        return foo


def interegular_to_wfsa(pattern, name=lambda x: x, charset='core'):
    if charset == 'core':
        charset = set(string.printable)
    elif isinstance(charset, set):
        pass
    else:
        # TODO: implement other charsets
        raise NotImplementedError(f'charset {charset} not implemented')

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
                            f'Excluding multi-character arc {A!r} in pattern {pattern!r} (possibly a result of case insensitivity of arcs {expand_alphabet(a)})'
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
                            f'Excluding multi-character arc {A!r} in pattern {pattern!r} (possibly a result of case insensitivity of arcs {expand_alphabet(a)})'
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
