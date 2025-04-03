import itertools
import re
from collections import Counter, defaultdict, namedtuple
from functools import cached_property
from itertools import product

# import nltk
from arsenal import Integerizer, colors

from genlm.grammar.fst import FST
from genlm.grammar.wfsa import EPSILON
from genlm.grammar.linear import WeightedGraph
from genlm.grammar.semiring import Boolean, Expectation, Float


def _gen_nt(prefix=""):
    """Generate a novel nonterminal symbol name.

    Args:
        prefix (str): Optional prefix for the generated symbol name. Defaults to ''.

    Returns:
        str: A unique nonterminal symbol name with format '{prefix}@{counter}'
    """
    _gen_nt.i += 1
    return f"{prefix}@{_gen_nt.i}"


_gen_nt.i = 0

Other = namedtuple("Other", "x")

NotNull = namedtuple("NotNull", "x")

Slash = namedtuple("Slash", "Y, Z, i")


class Rule:
    """A weighted production rule in a context-free grammar.

    Attributes:
        w: Weight of the rule
        head: Left-hand side nonterminal symbol
        body: Right-hand side sequence of symbols
    """

    def __init__(self, w, head, body):
        """Initialize a Rule.

        Args:
            w: Weight of the rule
            head: Left-hand side nonterminal symbol
            body: Right-hand side sequence of symbols
        """
        self.w = w
        self.head = head
        self.body = body
        self._hash = hash((head, body))

    def __eq__(self, other):
        return (
            isinstance(other, Rule)
            and self.w == other.w
            and self._hash == other._hash
            and other.head == self.head
            and other.body == self.body
        )

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"{self.w}: {self.head} → {' '.join(map(str, self.body))}"


class Derivation:
    """A derivation tree in a context-free grammar.

    Attributes:
        r (Rule): The rule used at this node, or None
        x: The symbol at this node
        ys: Child nodes of the derivation
    """

    def __init__(self, r, x, *ys):
        """Initialize a Derivation.

        Args:
            r (Rule): The rule used at this node, or None
            x: The symbol at this node
            *ys: Child nodes of the derivation
        """
        assert isinstance(r, Rule) or r is None
        self.r = r
        self.x = x
        self.ys = ys

    def __hash__(self):
        return hash((self.r, self.x, self.ys))

    def __eq__(self, other):
        return (self.r, self.x, self.ys) == (other.r, other.x, other.ys)

    def __repr__(self):
        open_p = colors.dark.white % "("
        close_p = colors.dark.white % ")"
        children = " ".join(str(y) for y in self.ys)
        return f"{open_p}{self.x} {children}{close_p}"

    def weight(self):
        """Compute the weight of this derivation.

        Returns:
            The weight of this derivation, computed by multiplying the rule weight
            with the weights of all child derivations.
        """
        W = self.r.w
        for y in self.ys:
            if isinstance(y, Derivation):
                W *= y.weight()
        return W

    def Yield(self):
        """Return the yield (terminal string) of this derivation.

        Returns:
            tuple: The sequence of terminal symbols at the leaves of this derivation tree.
        """
        if isinstance(self, Derivation):
            return tuple(w for y in self.ys for w in Derivation.Yield(y))
        else:
            return (self,)

    def _repr_html_(self):
        return self.to_nltk()._repr_svg_()


class CFG:
    """
    Weighted Context-free Grammar

    A weighted context-free grammar consists of:\n
    - `R`: A semiring that defines the weights\n
    - `S`: A start symbol (nonterminal)\n
    - `V`: A set of terminal symbols (vocabulary)\n
    - `N`: A set of nonterminal symbols\n
    - `rules`: A list of weighted production rules\n

    Each rule has the form: w: X -> Y1 Y2 ... Yn where:\n
    - w is a weight from the semiring R\n
    - X is a nonterminal symbol\n
    - Y1...Yn are terminal or nonterminal symbols\n
    """

    def __init__(self, R, S, V):
        """
        Initialize a weighted CFG.

        Args:
            R: The semiring for rule weights
            S: The start symbol (nonterminal)
            V: The set of terminal symbols (vocabulary)
        """
        self.R = R  # semiring
        self.V = V  # alphabet
        self.N = {S}  # nonterminals
        self.S = S  # unique start symbol
        self.rules = []  # rules
        self._trim_cache = [None, None]

    def __repr__(self):
        """Return string representation of the grammar."""
        return "Grammar {\n%s\n}" % "\n".join(f"  {r}" for r in self)

    def _repr_html_(self):
        """Return HTML representation of the grammar for Jupyter notebooks."""
        return f'<pre style="width: fit-content; text-align: left; border: thin solid black; padding: 0.5em;">{self}</pre>'

    @classmethod
    def from_string(
        cls,
        string,
        semiring,
        comment="#",
        start="S",
        is_terminal=lambda x: not x[0].isupper(),
    ):
        """
        Create a CFG from a string representation.

        Args:
            string: The grammar rules as a string
            semiring: The semiring for rule weights
            comment: Comment character to ignore lines (default: '#')
            start: Start symbol (default: 'S')
            is_terminal: Function to identify terminal symbols (default: lowercase first letter)

        Returns:
            A new CFG instance
        """
        V = set()
        cfg = cls(R=semiring, S=start, V=V)
        string = string.replace("->", "→")  # synonym for the arrow
        for line in string.split("\n"):
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            try:
                [(w, lhs, rhs)] = re.findall(r"(.*):\s*(\S+)\s*→\s*(.*)$", line)
                lhs = lhs.strip()
                rhs = rhs.strip().split()
                for x in rhs:
                    if is_terminal(x):
                        V.add(x)
                cfg.add(semiring.from_string(w), lhs, *rhs)
            except ValueError:
                raise ValueError(f"bad input line:\n{line}")  # pylint: disable=W0707
        return cfg

    def __getitem__(self, root):
        """
        Return a grammar that denotes the sublanguage of the nonterminal `root`.

        Args:
            root: The nonterminal to use as the new start symbol

        Returns:
            A new CFG with root as the start symbol
        """
        new = self.spawn(S=root)
        for r in self:
            new.add(r.w, r.head, *r.body)
        return new

    def __len__(self):
        """Return number of rules in the grammar."""
        return len(self.rules)

    def __call__(self, xs):
        """
        Compute the total weight of the sequence xs.

        Args:
            xs: A sequence of terminal symbols

        Returns:
            The total weight of all derivations of xs
        """
        self = self.cnf  # need to do this here because the start symbol might change
        return self._parse_chart(xs)[0, self.S, len(xs)]

    def _parse_chart(self, xs):
        """
        Implements CKY algorithm for evaluating the total weight of the xs sequence.

        Args:
            xs: A sequence of terminal symbols

        Returns:
            A chart containing the weights of all subderivations
        """
        (nullary, terminal, binary) = self._cnf  # will convert to CNF
        N = len(xs)
        # nullary rule
        c = self.R.chart()
        for i in range(N + 1):
            c[i, self.S, i] += nullary
        # preterminal rules
        for i in range(N):
            for r in terminal[xs[i]]:
                c[i, r.head, i + 1] += r.w
        # binary rules
        for span in range(2, N + 1):
            for i in range(N - span + 1):
                k = i + span
                for j in range(i + 1, k):
                    for r in binary:
                        X, [Y, Z] = r.head, r.body
                        c[i, X, k] += r.w * c[i, Y, j] * c[j, Z, k]
        return c

    def language(self, depth):
        """
        Enumerate strings generated by this cfg by derivations up to the given depth.

        Args:
            depth: Maximum derivation depth to consider

        Returns:
            A chart containing the weighted language up to the given depth
        """
        lang = self.R.chart()
        for d in self.derivations(self.S, depth):
            lang[d.Yield()] += d.weight()
        return lang

    @cached_property
    def rhs(self):
        """
        Map from each nonterminal to the list of rules with it as their left-hand side.

        Returns:
            A dict mapping nonterminals to lists of rules
        """
        rhs = defaultdict(list)
        for r in self:
            rhs[r.head].append(r)
        return rhs

    def is_terminal(self, x):
        """Return True if x is a terminal symbol."""
        return x in self.V

    def is_nonterminal(self, X):
        """Return True if X is a nonterminal symbol."""
        return not self.is_terminal(X)

    def __iter__(self):
        """Iterate over the rules in the grammar."""
        return iter(self.rules)

    @property
    def size(self):
        """Return total size of the grammar (sum of rule lengths)."""
        return sum(1 + len(r.body) for r in self)

    @property
    def num_rules(self):
        """Return number of rules in the grammar."""
        return len(self.rules)

    @property
    def expected_length(self):
        """
        Compute the expected length of a string using the Expectation semiring.

        Returns:
            The expected length of strings generated by this grammar

        Raises:
            AssertionError: If grammar is not over the Float semiring
        """
        assert self.R == Float, (
            "This method only supports grammars over the Float semiring"
        )
        new_cfg = self.__class__(R=Expectation, S=self.S, V=self.V)
        for r in self:
            new_cfg.add(
                Expectation(r.w, r.w * sum(self.is_terminal(y) for y in r.body)),
                r.head,
                *r.body,
            )
        return new_cfg.treesum().score[1]

    def spawn(self, *, R=None, S=None, V=None):
        """
        Create an empty grammar with the same R, S, and V.

        Args:
            R: Optional new semiring
            S: Optional new start symbol
            V: Optional new vocabulary

        Returns:
            A new empty CFG with specified parameters
        """
        return self.__class__(
            R=self.R if R is None else R,
            S=self.S if S is None else S,
            V=set(self.V) if V is None else V,
        )

    def add(self, w, head, *body):
        """
        Add a rule of the form w: head -> body1, body2, ... body_k.

        Args:
            w: The rule weight
            head: The left-hand side nonterminal
            *body: The right-hand side symbols

        Returns:
            The added rule, or None if weight is zero
        """
        if w == self.R.zero:
            return  # skip rules with weight zero
        self.N.add(head)
        r = Rule(w, head, body)
        self.rules.append(r)
        return r

    def renumber(self):
        """
        Rename nonterminals to integers.

        Returns:
            A new CFG with integer nonterminals
        """
        i = Integerizer()
        max_v = max((x for x in self.V if isinstance(x, int)), default=0)
        return self.rename(lambda x: i(x) + max_v + 1)

    def rename(self, f):
        """
        Return a new grammar that is the result of applying f to each nonterminal.

        Args:
            f: Function to rename nonterminals

        Returns:
            A new CFG with renamed nonterminals
        """
        new = self.spawn(S=f(self.S))
        for r in self:
            new.add(
                r.w, f(r.head), *((y if self.is_terminal(y) else f(y) for y in r.body))
            )
        return new

    def map_values(self, f, R):
        """
        Return a new grammar that is the result of applying f: self.R -> R to each rule's weight.

        Args:
            f: Function to map weights
            R: New semiring for weights

        Returns:
            A new CFG with mapped weights
        """
        new = self.spawn(R=R)
        for r in self:
            new.add(f(r.w), r.head, *r.body)
        return new

    def assert_equal(self, other, verbose=False, throw=True):
        """
        Assertion for the equality of self and other modulo rule reordering.

        Args:
            other: The grammar to compare against
            verbose: If True, print differences
            throw: If True, raise AssertionError on inequality

        Raises:
            AssertionError: If grammars are not equal and throw=True
        """
        assert verbose or throw
        if isinstance(other, str):
            other = self.__class__.from_string(other, self.R)
        if verbose:
            # TODO: need to check the weights in the print out; we do it in the assertion
            S = set(self.rules)
            G = set(other.rules)
            for r in sorted(S | G, key=str):
                if r in S and r in G:
                    continue
                # if r in S and r not in G: continue
                # if r not in S and r in G: continue
                print(
                    colors.mark(r in S),
                    # colors.mark(r in S and r in G),
                    colors.mark(r in G),
                    r,
                )
        assert not throw or Counter(self.rules) == Counter(other.rules), (
            f"\n\nhave=\n{str(self)}\nwant=\n{str(other)}"
        )

    def treesum(self, **kwargs):
        """
        Total weight of the start symbol.

        Returns:
            The total weight of all derivations from the start symbol
        """
        return self.agenda(**kwargs)[self.S]

    def trim(self, bottomup_only=False):
        """
        Return an equivalent grammar with no dead or useless nonterminals or rules.

        Args:
            bottomup_only: If True, only remove non-generating nonterminals

        Returns:
            A new trimmed CFG
        """
        if self._trim_cache[bottomup_only] is not None:
            return self._trim_cache[bottomup_only]

        C = set(self.V)
        C.update(e.head for e in self.rules if len(e.body) == 0)

        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for e in self:
            incoming[e.head].append(e)
            for b in e.body:
                outgoing[b].append(e)

        agenda = set(C)
        while agenda:
            x = agenda.pop()
            for e in outgoing[x]:
                if all((b in C) for b in e.body):
                    if e.head not in C:
                        C.add(e.head)
                        agenda.add(e.head)

        if bottomup_only:
            val = self._trim(C)
            self._trim_cache[bottomup_only] = val
            val._trim_cache[bottomup_only] = val
            return val

        T = {self.S}
        agenda.update(T)
        while agenda:
            x = agenda.pop()
            for e in incoming[x]:
                # assert e.head in T
                for b in e.body:
                    if b not in T and b in C:
                        T.add(b)
                        agenda.add(b)

        val = self._trim(T)
        self._trim_cache[bottomup_only] = val
        val._trim_cache[bottomup_only] = val
        return val

    def cotrim(self):
        """
        Trim the grammar so that all nonterminals are generating.

        Returns:
            A new CFG with only generating nonterminals
        """
        return self.trim(bottomup_only=True)

    def _trim(self, symbols):
        """
        Helper method for trim() - creates new grammar with only given symbols.

        Args:
            symbols: Set of symbols to keep

        Returns:
            A new CFG with only rules using the given symbols
        """
        new = self.spawn()
        for p in self:
            if p.head in symbols and p.w != self.R.zero and set(p.body) <= symbols:
                new.add(p.w, p.head, *p.body)
        return new

    # ___________________________________________________________________________
    # Derivation enumeration

    def derivations(self, X, H):
        """
        Enumerate derivations of symbol X with height <= H.

        Args:
            X: The symbol to derive from (default: start symbol)
            H: Maximum derivation height

        Yields:
            Derivation objects representing derivation trees
        """
        if X is None:
            X = self.S
        if self.is_terminal(X):
            yield X
        elif H <= 0:
            return
        else:
            for r in self.rhs[X]:
                for ys in self._derivations_list(r.body, H - 1):
                    yield Derivation(r, X, *ys)

    def _derivations_list(self, Xs, H):
        """
        Helper method for derivations; expands any list of symbols X up to depth H.

        Args:
            Xs: List of symbols to derive
            H: Maximum derivation height

        Yields:
            Tuples of derivations
        """
        if len(Xs) == 0:
            yield ()
        else:
            for x in self.derivations(Xs[0], H):
                for xs in self._derivations_list(Xs[1:], H):
                    yield (x, *xs)

    # ___________________________________________________________________________
    # Transformations

    def _unary_graph(self):
        """
        Compute the matrix closure of unary rules.

        Returns:
            A WeightedGraph representing unary rule closure
        """
        A = WeightedGraph(self.R)
        for r in self:
            if len(r.body) == 1 and self.is_nonterminal(r.body[0]):
                A[r.head, r.body[0]] += r.w
        A.N |= self.N
        return A

    def _unary_graph_transpose(self):
        """
        Compute the matrix closure of unary rules (transposed).

        Returns:
            A WeightedGraph representing transposed unary rule closure
        """
        A = WeightedGraph(self.R)
        for r in self:
            if len(r.body) == 1 and self.is_nonterminal(r.body[0]):
                A[r.body[0], r.head] += r.w
        A.N |= self.N
        return A

    def unaryremove(self):
        """
        Return an equivalent grammar with no unary rules.

        Returns:
            A new CFG without unary rules
        """
        W = self._unary_graph().closure_scc_based()
        # W = self._unary_graph().closure_reference()

        new = self.spawn()
        for r in self:
            if len(r.body) == 1 and self.is_nonterminal(r.body[0]):
                continue
            for Y in self.N:
                new.add(W[Y, r.head] * r.w, Y, *r.body)

        return new

    def has_unary_cycle(self):
        """
        Check if the grammar has unary cycles.

        Returns:
            True if the grammar contains unary cycles
        """
        f = self._unary_graph().buckets
        return any(
            True for r in self if len(r.body) == 1 and f.get(r.head) == f.get(r.body[0])
        )

    def unarycycleremove(self, trim=True):
        """
        Return an equivalent grammar with no unary cycles.

        Args:
            trim: If True, trim the resulting grammar

        Returns:
            A new CFG without unary cycles
        """

        def bot(x):
            return x if x in acyclic else (x, "bot")

        G = self._unary_graph()

        new = self.spawn(S=self.S)

        bucket = G.buckets

        acyclic = set()
        for nodes, _ in G.Blocks:
            if len(nodes) == 1:
                [X] = nodes
                if G[X, X] == self.R.zero:
                    acyclic.add(X)

        # run Lehmann's on each cylical SCC
        for nodes, W in G.Blocks:
            if len(nodes) == 1:
                [X] = nodes
                if X in acyclic:
                    continue

            for X1, X2 in W:
                new.add(W[X1, X2], X1, bot(X2))

        for r in self:
            if len(r.body) == 1 and bucket.get(r.body[0]) == bucket[r.head]:
                continue
            new.add(r.w, bot(r.head), *r.body)

        # TODO: figure out how to ensure that the new grammar is trimmed by
        # construction (assuming the input grammar was trim).
        if trim:
            new = new.trim()

        return new

    def nullaryremove(self, binarize=True, trim=True, **kwargs):
        """
        Return an equivalent grammar with no nullary rules except for one at the start symbol.

        Args:
            binarize: If True, binarize the grammar first
            trim: If True, trim the resulting grammar
            **kwargs: Additional arguments passed to _push_null_weights

        Returns:
            A new CFG without nullary rules (except at start)
        """
        # A really wide rule can take a very long time because of the power set
        # in this rule so it is really important to binarize.
        if binarize:
            self = self.binarize()  # pragma: no cover
        self = self.separate_start()
        tmp = self._push_null_weights(self.null_weight(), **kwargs)
        return tmp.trim() if trim else tmp

    def null_weight(self):
        """
        Compute the map from nonterminal to total weight of generating the empty string.

        Returns:
            A dict mapping nonterminals to their null weights
        """
        ecfg = self.spawn(V=set())
        for p in self:
            if not any(self.is_terminal(y) for y in p.body):
                ecfg.add(p.w, p.head, *p.body)
        return ecfg.agenda()

    def null_weight_start(self):
        """
        Compute the null weight of the start symbol.

        Returns:
            The total weight of generating the empty string from the start symbol
        """
        return self.null_weight()[self.S]

    def _push_null_weights(self, null_weight, rename=NotNull):
        """
        Returns a grammar that generates the same weighted language but is nullary-free
        at all nonterminals except its start symbol.

        Args:
            null_weight: Dict mapping nonterminals to their null weights
            rename: Function to rename nonterminals (default: NotNull)

        Returns:
            A new CFG without nullary rules (except at start)
        """
        # Warning: this method might have issues when `separate_start` hasn't
        # been run before.  So we run it rather than leaving it up to chance.
        assert self.S not in {y for r in self for y in r.body}

        def f(x):
            "Rename nonterminal if necessary"
            if (
                null_weight[x] == self.R.zero or x == self.S
            ):  # not necessary; keep old name
                return x
            else:
                return rename(x)

        rcfg = self.spawn()
        rcfg.add(null_weight[self.S], self.S)

        for r in self:
            if len(r.body) == 0:
                continue  # drop nullary rule

            for B in product([0, 1], repeat=len(r.body)):
                v, new_body = r.w, []

                for i, b in enumerate(B):
                    if b:
                        v *= null_weight[r.body[i]]
                    else:
                        new_body.append(f(r.body[i]))

                # exclude the cases that would be new nullary rules!
                if len(new_body) > 0:
                    rcfg.add(v, f(r.head), *new_body)

        return rcfg

    def separate_start(self):
        """
        Ensure that the start symbol does not appear on the RHS of any rule.

        Returns:
            A new CFG with start symbol only on LHS
        """
        # create a new start symbol if the current one appears on the rhs of any existing rule
        if self.S in {y for r in self for y in r.body}:
            S = _gen_nt(self.S)
            new = self.spawn(S=S)
            # preterminal rules
            new.add(self.R.one, S, self.S)
            for r in self:
                new.add(r.w, r.head, *r.body)
            return new
        else:
            return self

    def separate_terminals(self):
        """
        Ensure that each terminal is produced by a preterminal rule.

        Returns:
            A new CFG with terminals only in preterminal rules
        """
        one = self.R.one
        new = self.spawn()

        _preterminal = {}

        def preterminal(x):
            y = _preterminal.get(x)
            if y is None:
                y = new.add(one, _gen_nt(), x)
                _preterminal[x] = y
            return y

        for r in self:
            if len(r.body) == 1 and self.is_terminal(r.body[0]):
                new.add(r.w, r.head, *r.body)
            else:
                new.add(
                    r.w,
                    r.head,
                    *(
                        (preterminal(y).head if self.is_terminal(y) else y)
                        for y in r.body
                    ),
                )

        return new

    def binarize(self):
        """
        Return an equivalent grammar with arity ≤ 2.

        Returns:
            A new CFG with binary rules
        """
        new = self.spawn()

        stack = list(self)
        while stack:
            p = stack.pop()
            if len(p.body) <= 2:
                new.add(p.w, p.head, *p.body)
            else:
                stack.extend(self._fold(p, [(0, 1)]))

        return new

    def _fold(self, p, I):
        """
        Helper method for binarization - folds a rule into binary rules.

        Args:
            p: The rule to fold
            I: List of (start,end) indices for folding

        Returns:
            List of new binary rules
        """
        # new productions
        P, heads = [], []
        for i, j in I:
            head = _gen_nt()
            heads.append(head)
            body = p.body[i : j + 1]
            P.append(Rule(self.R.one, head, body))

        # new "head" production
        body = tuple()
        start = 0
        for (end, n), head in zip(I, heads):
            body += p.body[start:end] + (head,)
            start = n + 1
        body += p.body[start:]
        P.append(Rule(p.w, p.head, body))

        return P

    @cached_property
    def cnf(self):
        """
        Transform this grammar into Chomsky Normal Form (CNF).

        Returns:
            A new CFG in CNF
        """
        new = (
            self.separate_terminals()
            .nullaryremove(binarize=True)
            .trim()
            .unaryremove()
            .trim()
        )
        assert new.in_cnf(), "\n".join(
            str(r) for r in new._find_invalid_cnf_rule()
        )  # pragma: no cover
        return new

    # TODO: make CNF grammars a speciazed subclass of CFG.
    @cached_property
    def _cnf(self):
        """
        Note: Throws an exception if the grammar is not in CNF.

        Returns:
            Tuple of (nullary weight, terminal rules dict, binary rules list)
        """
        nullary = self.R.zero
        terminal = defaultdict(list)
        binary = []
        for r in self:
            if len(r.body) == 0:
                nullary += r.w
                assert r.head == self.S, [self.S, r]
            elif len(r.body) == 1:
                terminal[r.body[0]].append(r)
                assert self.is_terminal(r.body[0])
            else:
                assert len(r.body) == 2
                binary.append(r)
                assert self.is_nonterminal(r.body[0])
                assert self.is_nonterminal(r.body[1])
        return (nullary, terminal, binary)

    def in_cnf(self):
        """
        Return true if the grammar is in CNF.

        Returns:
            True if grammar is in Chomsky Normal Form
        """
        return len(list(self._find_invalid_cnf_rule())) == 0

    def _find_invalid_cnf_rule(self):
        """
        Return true if the grammar is in CNF.

        Yields:
            Rules that violate CNF
        """
        for r in self:
            assert r.head in self.N
            if len(r.body) == 0 and r.head == self.S:
                continue
            elif len(r.body) == 1 and self.is_terminal(r.body[0]):
                continue
            elif len(r.body) == 2 and all(
                self.is_nonterminal(y) and y != self.S for y in r.body
            ):
                continue
            else:
                yield r

    #    def has_nullary(self):
    #        return any((len(p.body) == 0) for p in self if p.head != self.S)

    def unfold(self, i, k):
        """
        Apply the unfolding transformation to rule i and subgoal k.

        Args:
            i: Index of rule to unfold
            k: Index of subgoal in rule body

        Returns:
            A new CFG with the rule unfolded
        """
        assert isinstance(i, int) and isinstance(k, int)
        s = self.rules[i]
        assert self.is_nonterminal(s.body[k])

        new = self.spawn()
        for j, r in enumerate(self):
            if j != i:
                new.add(r.w, r.head, *r.body)

        for r in self.rhs[s.body[k]]:
            new.add(s.w * r.w, s.head, *s.body[:k], *r.body, *s.body[k + 1 :])

        return new

    def dependency_graph(self):
        """
        Head-to-body dependency graph of the rules of the grammar.

        Returns:
            A WeightedGraph representing dependencies between symbols
        """
        deps = WeightedGraph(Boolean)
        for r in self:
            for y in r.body:
                deps[r.head, y] += Boolean.one
        deps.N |= self.N
        deps.N |= self.V
        return deps

    # TODO: the default treesum algorithm should probably be SCC-decomposed newton's method
    # def agenda(self, tol=1e-12, maxiter=float('inf')):
    def agenda(self, tol=1e-12, maxiter=100_000):
        """
        Agenda-based semi-naive evaluation for treesums.

        Args:
            tol: Convergence tolerance
            maxiter: Maximum iterations

        Returns:
            A chart containing the treesum weights
        """
        old = self.R.chart()

        # precompute the mapping from updates to where they need to go
        routing = defaultdict(list)
        for r in self:
            for k in range(len(r.body)):
                routing[r.body[k]].append((r, k))

        deps = self.dependency_graph()
        blocks = deps.blocks
        bucket = deps.buckets

        # helper function
        def update(x, W):
            change[bucket[x]][x] += W

        change = defaultdict(self.R.chart)
        for a in self.V:
            update(a, self.R.one)

        for r in self:
            if len(r.body) == 0:
                update(r.head, r.w)

        b = len(blocks)
        iteration = 0
        while b >= 0:
            iteration += 1

            # Move on to the next block
            if len(change[b]) == 0 or iteration > maxiter:
                b -= 1
                iteration = 0  # reset iteration number for the next bucket
                continue

            u, v = change[b].popitem()

            new = old[u] + v

            if self.R.metric(old[u], new) <= tol:
                continue

            for r, k in routing[u]:
                W = r.w
                for j in range(len(r.body)):
                    if u == r.body[j]:
                        if j < k:
                            W *= new
                        elif j == k:
                            W *= v
                        else:
                            W *= old[u]
                    else:
                        W *= old[r.body[j]]

                update(r.head, W)

            old[u] = new

        return old

    def naive_bottom_up(self, *, tol=1e-12, timeout=100_000):
        "Naive bottom-up evaluation for treesums; better to use `agenda`."

        def _approx_equal(U, V):
            return all((self.R.metric(U[X], V[X]) <= tol) for X in self.N)

        R = self.R
        V = R.chart()
        counter = 0
        while counter < timeout:
            U = self._bottom_up_step(V)
            if _approx_equal(U, V):
                break
            V = U
            counter += 1
        return V

    def _bottom_up_step(self, V):
        R = self.R
        one = R.one
        U = R.chart()
        for a in self.V:
            U[a] = one
        for p in self:
            update = p.w
            for X in p.body:
                if self.is_nonterminal(X):
                    update *= V[X]
            U[p.head] += update
        return U

    def prefix_weight(self, xs):
        "Total weight of all derivations that have `xs` as a prefix."
        return self.prefix_grammar(xs)

    @cached_property
    def prefix_grammar(self):
        "Grammar that generates prefixing of this grammar's language."
        return self @ prefix_transducer(self.R, self.V)

    def derivatives(self, s):
        "Return the sequence of derivatives for each prefix of `s`."
        M = len(s)
        D = [self]
        for m in range(M):
            D.append(D[m].derivative(s[m]))
        return D

    # Implementation note: This implementation of the derivative grammar
    # performs nullary elimination at the same time.
    def derivative(self, a, i=0):
        "Return a grammar that generates the derivative with respect to `a`."

        def slash(x, y):
            return Slash(x, y, i=i)

        D = self.spawn(S=slash(self.S, a))
        U = self.null_weight()
        for r in self:
            D.add(r.w, r.head, *r.body)
            delta = self.R.one
            for k, y in enumerate(r.body):
                if slash(r.head, a) in self.N:
                    continue  # SKIP!
                if self.is_terminal(y):
                    if y == a:
                        D.add(delta * r.w, slash(r.head, a), *r.body[k + 1 :])
                else:
                    D.add(
                        delta * r.w,
                        slash(r.head, a),
                        slash(r.body[k], a),
                        *r.body[k + 1 :],
                    )
                delta *= U[y]
        return D

    def _compose_bottom_up_epsilon(self, fst):
        "Determine which items of the composition grammar are supported"

        A = set()

        I = defaultdict(set)  # incomplete items
        C = defaultdict(set)  # complete items
        R = defaultdict(set)  # rules indexed by first subgoal; non-nullary

        special_rules = [Rule(self.R.one, a, (EPSILON, a)) for a in self.V] + [
            Rule(self.R.one, Other(self.S), (self.S,)),
            Rule(self.R.one, Other(self.S), (Other(self.S), EPSILON)),
        ]

        for r in itertools.chain(self, special_rules):
            if len(r.body) > 0:
                R[r.body[0]].add(r)

        # we have two base cases:
        #
        # base case 1: arcs
        for i, (a, _), j, _ in fst.arcs():
            A.add((i, a, (), j))  # empty tuple -> the rule 'complete'

        # base case 2: nullary rules
        for r in self:
            if len(r.body) == 0:
                for i in fst.states:
                    A.add((i, r.head, (), i))

        # drain the agenda
        while A:
            (i, X, Ys, j) = A.pop()

            # No pending items ==> the item is complete
            if not Ys:
                if j in C[i, X]:
                    continue
                C[i, X].add(j)

                # combine the newly completed item with incomplete rules that are
                # looking for an item like this one
                for h, X1, Zs in I[i, X]:
                    A.add((h, X1, Zs[1:], j))

                # initialize rules that can start with an item like this one
                for r in R[X]:
                    A.add((i, r.head, r.body[1:], j))

            # Still have pending items ==> advanced the pending items
            else:
                if (i, X, Ys) in I[j, Ys[0]]:
                    continue
                I[j, Ys[0]].add((i, X, Ys))

                for k in C[j, Ys[0]]:
                    A.add((i, X, Ys[1:], k))

        return C

    def __matmul__(self, fst):
        "Return a CFG denoting the pointwise product or composition of `self` and `fs`."

        # coerce something sequence like into a diagonal FST
        if isinstance(fst, (str, tuple)):
            fst = FST.from_string(fst, self.R)
        # coerce something FSA-like into an FST, might throw an error
        if not isinstance(fst, FST):
            fst = fst.to_fst()

        # Initialize the new CFG:
        # - its start symbol is chosen arbitrarily to be `self.S`
        # - its the alphabet changes - it is now 'output' alphabet of the transducer
        new_start = self.S
        new = self.spawn(S=new_start, V=fst.B - {EPSILON})

        # The bottom-up intersection algorithm is a two-pass algorithm
        #
        # Pass 1: Determine the set of items that are possiblly nonzero-valued
        C = self._compose_bottom_up_epsilon(fst)

        special_rules = [Rule(self.R.one, a, (EPSILON, a)) for a in self.V] + [
            Rule(self.R.one, Other(self.S), (self.S,)),
            Rule(self.R.one, Other(self.S), (Other(self.S), EPSILON)),
        ]

        def join(start, Ys):
            """
            Helper method; expands the rule body

            Given Ys = [Y_1, ... Y_K], we will enumerate expansion of the form

            (s_0, Y_1, s_1), (s_1, Y_2, s_2), ..., (s_{k-1}, Y_K, s_K)

            where each (s_k, Y_k, s_k) in the expansion is a completed items
            (i.e., \forall k: (s_k, Y_k, s_k) in C).
            """
            if not Ys:
                yield []
            else:
                for K in C[start, Ys[0]]:
                    for rest in join(K, Ys[1:]):
                        yield [(start, Ys[0], K)] + rest

        start = {I for (I, _) in C}

        for r in itertools.chain(self, special_rules):
            if len(r.body) == 0:
                for s in fst.states:
                    new.add(r.w, (s, r.head, s))
            else:
                for I in start:
                    for rhs in join(I, r.body):
                        K = rhs[-1][-1]
                        new.add(r.w, (I, r.head, K), *rhs)

        for i, wi in fst.start.items():
            for k, wf in fst.stop.items():
                new.add(wi * wf, new_start, (i, Other(self.S), k))

        for i, (a, b), j, w in fst.arcs():
            if b == EPSILON:
                new.add(w, (i, a, j))
            else:
                new.add(w, (i, a, j), b)
        return new

    def truncate_length(self, max_length):
        "Transform this grammar so that it only generates strings with length ≤ `max_length`."
        from genlm.grammar import WFSA

        m = WFSA(self.R)
        m.add_I(0, self.R.one)
        m.add_F(0, self.R.one)
        for t in range(max_length):
            for x in self.V:
                m.add_arc(t, x, t + 1, self.R.one)
            m.add_F(t + 1, self.R.one)
        return self @ m

    def materialize(self, max_length):
        "Return a `Chart` with this grammar's weighted language for strings ≤ `max_length`."
        return self.cnf.language(max_length).filter(lambda x: len(x) <= max_length)

    def to_bytes(self):
        """Convert terminal symbols from strings to bytes representation.

        This method creates a new grammar where all terminal string symbols are
        converted to their UTF-8 byte representation. Non-terminal symbols are
        preserved as-is.

        Returns:
            CFG: A new grammar with byte terminal symbols

        Raises:
            ValueError: If a terminal symbol is not a string
        """
        new = self.spawn(S=self.S, R=self.R, V=set())

        for r in self:
            new_body = []
            for x in r.body:
                if self.is_terminal(x):
                    if not isinstance(x, str):
                        raise ValueError(f"unsupported terminal type: {type(x)}")
                    bs = list(x.encode("utf-8"))
                    for b in bs:
                        new.V.add(b)
                    new_body.extend(bs)
                else:
                    new_body.append(x)
            new.add(r.w, r.head, *new_body)

        return new


def prefix_transducer(R, V):
    "Construct the prefix transducer over semiring `R` and alphabet `V`."
    P = FST(R)
    P.add_I(0, R.one)
    P.add_I(1, R.one)
    for x in V:
        P.add_arc(0, (x, x), 0, R.one)
        P.add_arc(0, (x, x), 1, R.one)
        P.add_arc(1, (x, EPSILON), 1, R.one)
    P.add_F(1, R.one)
    return P
