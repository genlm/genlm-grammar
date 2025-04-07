import html
from collections import defaultdict
from functools import cached_property

from arsenal import Integerizer
from frozendict import frozendict
from graphviz import Digraph

from genlm.grammar.linear import WeightedGraph

# EPSILON = "ε"
EPSILON = ""


class WFSA:
    """
    Weighted finite-state automata
    """

    def __init__(self, R):
        self.R = R
        self.alphabet = set()
        self.states = set()
        self.delta = defaultdict(lambda: defaultdict(R.chart))
        self.start = R.chart()
        self.stop = R.chart()

    def __repr__(self):
        return f"{__class__.__name__}({self.dim} states)"

    def __str__(self):
        output = []
        output.append("{")
        for p in self.states:
            output.append(f"  {p} \t\t({self.start[p]}, {self.stop[p]})")
            for a, q, w in self.arcs(p):
                output.append(f"    {a}: {q}\t[{w}]")
        output.append("}")
        return "\n".join(output)

    @property
    def dim(self):
        return len(self.states)

    def add_state(self, q):
        self.states.add(q)

    def add_arc(self, i, a, j, w):
        self.add_state(i)
        self.add_state(j)
        self.alphabet.add(a)
        self.delta[i][a][j] += w

    def add_I(self, q, w):
        self.add_state(q)
        self.start[q] += w

    def add_F(self, q, w):
        self.add_state(q)
        self.stop[q] += w

    def set_arc(self, i, a, j, w):
        self.add_state(i)
        self.add_state(j)
        self.alphabet.add(a)
        self.delta[i][a][j] = w

    def set_I(self, q, w):
        self.add_state(q)
        self.start[q] = w

    def set_F(self, q, w):
        self.add_state(q)
        self.stop[q] = w

    @property
    def I(self):
        for q, w in self.start.items():
            if w != self.R.zero:
                yield q, w

    @property
    def F(self):
        for q, w in self.stop.items():
            if w != self.R.zero:
                yield q, w

    def arcs(self, i=None, a=None):
        if i is not None:
            if a is not None:
                for j, w in self.delta[i][a].items():
                    yield j, w
            else:
                for a, T in self.delta[i].items():
                    for j, w in T.items():
                        yield a, j, w
        else:
            if a is None:
                for i in self.delta:
                    for a, T in self.delta[i].items():
                        for j, w in T.items():
                            yield i, a, j, w
            else:
                raise NotImplementedError

    def rename(self, f):
        "Note: If `f` is not bijective, states may merge."
        m = self.spawn()
        for i, w in self.I:
            m.add_I(f(i), w)
        for i, w in self.F:
            m.add_F(f(i), w)
        for i, a, j, w in self.arcs():
            m.add_arc(f(i), a, f(j), w)
        return m

    def to_fst(self):
        from genlm.grammar.fst import FST

        return FST.diag(self)

    def rename_apart(self, other):
        f = Integerizer()
        return (self.rename(lambda i: f((0, i))), other.rename(lambda i: f((1, i))))

    @cached_property
    def renumber(self):
        return self.rename(Integerizer())

    def spawn(self, *, keep_init=False, keep_arcs=False, keep_stop=False):
        "Returns a new WFSA in the same semiring."
        m = self.__class__(self.R)
        if keep_init:
            for q, w in self.I:
                m.add_I(q, w)
        if keep_arcs:
            for i, a, j, w in self.arcs():
                m.add_arc(i, a, j, w)
        if keep_stop:
            for q, w in self.F:
                m.add_F(q, w)
        return m

    def __call__(self, xs):
        self = self.epsremove
        prev = self.start
        for x in xs:
            curr = self.R.chart()
            for i in prev:
                for j, w in self.arcs(i, x):
                    curr[j] += prev[i] * w
            prev = curr
        total = self.R.zero
        for j, w in self.F:
            total += prev[j] * w
        return total

    @cached_property
    def E(self):
        "Weighted graph of epsilon arcs"
        E = WeightedGraph(self.R)
        for i, a, j, w in self.arcs():
            if a == EPSILON:
                E[i, j] += w
        E.N |= self.states
        return E

    @cached_property
    def epsremove(self):
        "Return an equivalent machine with no epsilon arcs."
        E = self.E
        S = E.closure()
        new = self.spawn(keep_stop=True)
        for i, w_i in self.I:
            for k in S.outgoing[i]:
                new.add_I(k, w_i * S[i, k])
        for i, a, j, w_ij in self.arcs():
            if a == EPSILON:
                continue
            for k in S.outgoing[j]:
                new.add_arc(i, a, k, w_ij * S[j, k])
        return new

    @cached_property
    def reverse(self):
        "creates a reversed machine"
        # create the new machine
        R = self.spawn()
        # reverse each arc
        for i, a, j, w in self.arcs():
            R.add_arc(j, a, i, w)  # pylint: disable=arguments-out-of-order
        # reverse initial and final states
        for i, w in self.F:
            R.add_I(i, w)
        for i, w in self.I:
            R.add_F(i, w)
        return R

    def __add__(self, other):
        self, other = self.rename_apart(other)
        U = self.spawn(keep_init=True, keep_arcs=True, keep_stop=True)
        # add arcs, initial and final states from argument
        for i, a, j, w in other.arcs():
            U.add_arc(i, a, j, w)
        for q, w in other.I:
            U.add_I(q, w)
        for q, w in other.F:
            U.add_F(q, w)
        return U

    #    def __sub__(self, other):
    #        "Assumes -w exists for all weights."
    #        self, other = self.rename_apart(other)
    #        U = self.spawn(keep_init=True, keep_arcs=True, keep_stop=True)
    #        # add arcs, initial and final states from argument
    #        for q, w in other.I:            U.add_I(q, -w)
    #        for i, a, j, w in other.arcs(): U.add_arc(i, a, j, w)
    #        for q, w in other.F:            U.add_F(q, w)
    #        return U

    def __mul__(self, other):
        #        if not isinstance(other, self.__class__): return other.__rmul__(self)

        self, other = self.rename_apart(other)
        C = self.spawn(keep_init=True, keep_arcs=True)
        # add arcs, initial and final states from argument
        for i, a, j, w in other.arcs():
            C.add_arc(i, a, j, w)
        for q, w in other.F:
            C.add_F(q, w)
        # connect the final states from `self` to initial states from `other`
        for i1, w1 in self.F:
            for i2, w2 in other.I:
                C.add_arc(i1, EPSILON, i2, w1 * w2)
        return C

    @property
    def zero(self):
        return self.__class__(self.R)

    @property
    def one(self):
        return self.__class__.lift(EPSILON, self.R.one)

    def star(self):
        return self.one + self.kleene_plus()

    def kleene_plus(self):
        "self^+"
        m = self.spawn(keep_init=True, keep_arcs=True, keep_stop=True)
        for i, w1 in self.F:
            for j, w2 in self.I:
                m.add_arc(i, EPSILON, j, w1 * w2)
        return m

    def _repr_svg_(self):
        return self.graphviz()._repr_image_svg_xml()

    def graphviz(
        self,
        fmt=str,
        fmt_node=lambda x: " ",
        fmt_edge=lambda i,
        a,
        j,
        w: f"{html.escape(str(':'.join(str(A or 'ε') for A in a)) if isinstance(a, tuple) else str(a))}/{w}",
    ):
        if len(self.states) == 0:
            import warnings

            warnings.warn("empty visualization")
        g = Digraph(
            graph_attr=dict(rankdir="LR"),
            node_attr=dict(
                fontname="Monospace",
                fontsize="10",
                height=".05",
                width=".05",
                # margin="0.055,0.042"
                margin="0,0",
            ),
            edge_attr=dict(
                # arrowhead='vee',
                arrowsize="0.3",
                fontname="Monospace",
                fontsize="9",
            ),
        )
        f = Integerizer()
        for i, w in self.I:
            start = f"<start_{i}>"
            g.node(start, label="", shape="point", height="0", width="0")
            g.edge(start, str(f(i)), label=f"{fmt(w)}")
        for i in self.states:
            g.node(str(f(i)), label=str(fmt_node(i)), shape="circle")
        for i, w in self.F:
            stop = f"<stop_{i}>"
            g.node(stop, label="", shape="point", height="0", width="0")
            g.edge(str(f(i)), stop, label=f"{fmt(w)}")
        # for i, a, j, w in sorted(self.arcs()):
        for i, a, j, w in self.arcs():
            g.edge(str(f(i)), str(f(j)), label=f"{fmt_edge(i, a, j, w)}")
        return g

    @classmethod
    def lift(cls, x, w, R=None):
        if R is None:
            R = w.__class__
        m = cls(R=R)
        m.add_I(0, R.one)
        m.add_arc(0, x, 1, w)
        m.add_F(1, R.one)
        return m

    @classmethod
    def from_string(cls, xs, R, w=None):
        m = cls(R)
        m.add_I(xs[:0], R.one)
        for i in range(len(xs)):
            m.add_arc(xs[:i], xs[i], xs[: i + 1], R.one)
        m.add_F(xs, (R.one if w is None else w))
        return m

    @classmethod
    def from_strings(cls, Xs, R):
        m = cls(R)
        for xs in Xs:
            m.set_I(xs[:0], R.one)
            for i in range(len(xs)):
                m.set_arc(xs[:i], xs[i], xs[: i + 1], R.one)
            m.set_F(xs, R.one)
        return m

    def total_weight(self):
        b = self.backward
        return sum(self.start[i] * b[i] for i in self.start)

    @cached_property
    def G(self):
        G = WeightedGraph(self.R)
        for i, _, j, w in self.arcs():
            G[i, j] += w
        G.N |= self.states
        return G

    @cached_property
    def K(self):
        return self.G.closure_scc_based()

    @cached_property
    def forward(self):
        return self.G.solve_left(self.start)

    @cached_property
    def backward(self):
        return self.G.solve_right(self.stop)

    @cached_property
    def push(self):
        "Weight pushing algorithm (Mohri, 2001); assumes v**-1 possible for weights."
        V = self.backward
        new = self.spawn()
        for i in self.states:
            if V[i] == self.R.zero:
                continue
            new.add_I(i, self.start[i] * V[i])
            new.add_F(i, V[i] ** (-1) * self.stop[i])
            for a, j, w in self.arcs(i):
                new.add_arc(i, a, j, V[i] ** (-1) * w * V[j])
        return new

    @cached_property
    def trim_vals(self):
        """
        This method provides fine-grained trimming based on semiring values rather
        than the coarser-grained boolean approximation provided by `trim`.
        However, it is generally slower to evaluation.
        """
        forward = self.forward
        backward = self.backward
        # determine the set of active state, (i.e., those with nonzero forward and backward weights)
        return self._trim(
            {
                i
                for i in self.states
                if forward[i] != self.R.zero and backward[i] != self.R.zero
            }
        )

    def accessible(self):
        stack = list(self.start)
        visited = set(self.start)
        while stack:
            P = stack.pop()
            for _, Q, _ in self.arcs(P):
                if Q not in visited:
                    stack.append(Q)
                    visited.add(Q)
        return visited

    def co_accessible(self):
        return self.reverse.accessible()

    @cached_property
    def trim(self):
        return self._trim(self.accessible() & self.co_accessible())

    def _trim(self, active):
        new = self.spawn()
        for i in active:
            new.add_I(i, self.start[i])
            new.add_F(i, self.stop[i])
            for a, j, w in self.arcs(i):
                if j in active:
                    new.add_arc(i, a, j, w)
        return new

    @cached_property
    def min_det(self):
        """
        Implements Brzozowski's_minimization algorithm.
        See https://ralphs16.github.io/src/CatLectures/HW_Brzozowski.pdf and
        https://link.springer.com/chapter/10.1007/978-3-642-39274-0_17.
        """
        return self.reverse.determinize.trim.reverse.determinize.trim

    @cached_property
    def determinize(self):
        """
        Mohri (2009)'s "on-the-fly" determinization semi-algorithm.
        https://link.springer.com/chapter/10.1007/978-3-642-01492-5_6

        Use with caution as this method may not terminate.
        """

        self = self.epsremove.push

        def _powerarcs(Q):
            U = {a: self.R.chart() for a in self.alphabet}

            for i, u in Q.items():
                for a, j, v in self.arcs(i):
                    U[a][j] += u * v

            for a in U:
                R = U[a]
                W = sum(R.values(), start=self.R.zero)

                if 0:
                    # If we cannot extract a common factor, then all of the arcs will have weight one
                    yield a, frozendict(R), self.R.one

                else:
                    yield a, frozendict({p: W ** (-1) * R[p] for p in R}), W

        D = self.spawn()

        stack = []
        visited = set()

        Q = frozendict({i: w for i, w in self.I})
        D.add_I(Q, self.R.one)
        stack.append(Q)
        visited.add(Q)

        while stack:
            P = stack.pop()
            for a, Q, w in _powerarcs(P):
                if Q not in visited:
                    stack.append(Q)
                    visited.add(Q)
                D.add_arc(P, a, Q, w)

        for Q in D.states:
            for q in Q:
                D.add_F(Q, Q[q] * self.stop[q])

        return D

    def to_cfg(self, S=None, recursion="right"):
        """
        Convert the WFSA to a WCFG with the same weighted language.

        The option `recursion` in {"left", "right"} specifies whether the WCFG
        should be left or right recursive.

        """
        from genlm.grammar.cfg import CFG, _gen_nt

        if S is None:
            S = _gen_nt()
        cfg = CFG(R=self.R, V=self.alphabet - {EPSILON}, S=S)

        if recursion == "right":
            # add production rule for initial states
            for i, w in self.I:
                cfg.add(w, S, i)

            # add production rule for final states
            for i, w in self.F:
                cfg.add(w, i)

            # add other production rules
            for i, a, j, w in self.arcs():
                if a == EPSILON:
                    cfg.add(w, i, j)
                else:
                    cfg.add(w, i, a, j)

        else:
            assert recursion == "left"

            # add production rule for final states
            for i, w in self.F:
                cfg.add(w, S, i)

            # add production rule for initial states
            for i, w in self.I:
                cfg.add(w, i)

            # add other production rules
            for i, a, j, w in self.arcs():
                if a == EPSILON:
                    cfg.add(w, j, i)
                else:
                    cfg.add(w, j, i, a)

        return cfg

    def to_bytes(self):
        # Can be optimized, currently creates more states than necessary
        # when multiple characters emanating from the same state share a byte prefix.
        byte_wfsa = self.spawn(keep_init=True, keep_stop=True)

        state_counter = 0

        def get_new_state():
            nonlocal state_counter
            state = f"_bytes{state_counter}"
            state_counter += 1
            return state

        for i, a, j, w in self.arcs():
            if a == EPSILON:
                byte_wfsa.add_arc(i, a, j, w)
            elif isinstance(a, str):
                bs = a.encode("utf-8")
                if len(bs) == 1:
                    byte_wfsa.add_arc(i, bs[0], j, w)
                else:  # Multi-byte transition
                    curr = get_new_state()
                    byte_wfsa.add_arc(i, bs[0], curr, self.R.one)
                    for b in bs[1:-1]:
                        next_state = get_new_state()
                        byte_wfsa.add_arc(curr, b, next_state, self.R.one)
                        curr = next_state
                    byte_wfsa.add_arc(curr, bs[-1], j, w)
            else:
                raise ValueError(f"Invalid arc label {a} for byte conversion")

        return byte_wfsa
