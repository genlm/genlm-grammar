import numpy as np
from arsenal import Integerizer

from collections import defaultdict

from arsenal.datastructures.heap import LocatorMaxHeap

from genlm.grammar.cfglm import EOS, add_EOS, locally_normalize
from genlm.grammar.lm import LM
from genlm.grammar.semiring import Float
from genlm.grammar.cfg import CFG


class EarleyLM(LM):
    def __init__(self, cfg):
        if EOS not in cfg.V:
            cfg = add_EOS(cfg)
        self.cfg = cfg  # Note: <- cfg before prefix transform & normalization!
        self.model = Earley(cfg.prefix_grammar)
        super().__init__(V=cfg.V, eos=EOS)

    def p_next(self, context):
        assert set(context) <= self.V, f"OOVs detected: {set(context) - self.V}"
        return self.model.next_token_weights(self.model.chart(context)).normalize()

    def clear_cache(self):
        self.model.clear_cache()

    @classmethod
    def from_string(cls, x, semiring=Float, **kwargs):
        return cls(locally_normalize(CFG.from_string(x, semiring), **kwargs))


class Column:
    __slots__ = ("k", "i_chart", "c_chart", "waiting_for", "Q")

    def __init__(self, k):
        self.k = k
        self.i_chart = {}
        self.c_chart = {}

        # Within column J, this datastructure maps nonterminals Y to a set of items
        #   Y => {(I, X, Ys) | phrase(I,X/[Y],J) ≠ 0}
        self.waiting_for = defaultdict(list)


class Earley:
    """
    Implements a semiring-weighted version Earley's algorithm that runs in $\mathcal{O}(N^3|G|)$ time.
    Note that nullary rules and unary chain cycles will be been removed, altering the
    set of derivation trees.
    """

    __slots__ = (
        "cfg",
        "order",
        "_chart",
        "V",
        "eos",
        "_initial_column",
        "R_outgoing",
        "rhs",
        "ORDER_MAX",
        "intern_Ys",
        "unit_Ys",
        "first_Ys",
        "rest_Ys",
    )

    def __init__(self, cfg):
        cfg = cfg.nullaryremove(binarize=True).unarycycleremove().renumber()
        self.cfg = cfg

        # cache of chart columns
        self._chart = {}

        # Topological ordering on the grammar symbols so that we process unary
        # rules in a topological order.
        self.order = cfg._unary_graph_transpose().buckets

        self.ORDER_MAX = 1 + max(self.order.values())

        # left-corner graph
        R_outgoing = defaultdict(set)
        for r in cfg:
            if len(r.body) == 0:
                continue
            A = r.head
            B = r.body[0]
            if cfg.is_terminal(B):
                continue
            R_outgoing[A].add(B)
        self.R_outgoing = R_outgoing

        # Integerize rule right-hand side states
        intern_Ys = Integerizer()
        assert intern_Ys(()) == 0

        for r in self.cfg:
            for p in range(len(r.body) + 1):
                intern_Ys.add(r.body[p:])

        self.intern_Ys = intern_Ys

        self.rhs = {}
        for X in self.cfg.N:
            self.rhs[X] = []
            for r in self.cfg.rhs[X]:
                if r.body == ():
                    continue
                self.rhs[X].append((r.w, intern_Ys(r.body)))

        self.first_Ys = np.zeros(len(intern_Ys), dtype=object)
        self.rest_Ys = np.zeros(len(intern_Ys), dtype=int)
        self.unit_Ys = np.zeros(len(intern_Ys), dtype=int)

        for Ys, code in list(self.intern_Ys.items()):
            self.unit_Ys[code] = len(Ys) == 1
            if len(Ys) > 0:
                self.first_Ys[code] = Ys[0]
                self.rest_Ys[code] = intern_Ys(Ys[1:])

        # self.generate_rust_test_case()

        col = Column(0)
        self.PREDICT(col)
        self._initial_column = col

    def clear_cache(self):
        self._chart.clear()

    def __call__(self, x):
        N = len(x)

        # return if empty string
        if N == 0:
            return sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ())

        # initialize bookkeeping structures
        self._chart[()] = [self._initial_column]

        cols = self.chart(x)

        value = cols[N].c_chart.get((0, self.cfg.S))
        return value if value is not None else self.cfg.R.zero

    def chart(self, x):
        x = tuple(x)
        c = self._chart.get(x)
        if c is None:
            self._chart[x] = c = self._compute_chart(x)
        return c

    def _compute_chart(self, x):
        if len(x) == 0:
            return [self._initial_column]
        else:
            chart = self.chart(x[:-1])
            last_chart = self.next_column(chart, x[-1])
            return chart + [
                last_chart
            ]  # TODO: avoid list addition here as it is not constant time!

    def next_column(self, prev_cols, token):
        prev_col = prev_cols[-1]
        next_col = Column(prev_cols[-1].k + 1)
        next_col_c_chart = next_col.c_chart
        prev_col_i_chart = prev_col.i_chart

        rest_Ys = self.rest_Ys
        _update = self._update

        Q = LocatorMaxHeap()

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        for item in prev_col.waiting_for[token]:
            (I, X, Ys) = item
            _update(next_col, Q, I, X, rest_Ys[Ys], prev_col_i_chart[item])

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
        while Q:
            jy = Q.pop()[0]
            (J, Y) = jy

            col_J = prev_cols[J]
            col_J_i_chart = col_J.i_chart
            y = next_col_c_chart[jy]
            for customer in col_J.waiting_for[Y]:
                (I, X, Ys) = customer
                _update(next_col, Q, I, X, rest_Ys[Ys], col_J_i_chart[customer] * y)

        self.PREDICT(next_col)

        return next_col

    def PREDICT(self, col):
        # PREDICT: phrase(K, X/Ys, K) += rule(X -> Ys) with some filtering heuristics
        k = col.k

        # Filtering heuristic: Don't create the predicted item (K, X, [...], K)
        # unless there exists an item that wants the X item that it may
        # eventually provide.  In other words, for predicting this item to be
        # useful there must be an item of the form (I, X', [X, ...], K) in this
        # column for which lc(X', X) is true.
        if col.k == 0:
            agenda = [self.cfg.S]
        else:
            agenda = list(col.waiting_for)

        outgoing = self.R_outgoing

        reachable = set(agenda)

        while agenda:
            X = agenda.pop()
            for Y in outgoing[X]:
                if Y not in reachable:
                    reachable.add(Y)
                    agenda.append(Y)

        rhs = self.rhs
        _update = self._update
        for X in reachable:
            for w, Ys in rhs.get(X, ()):
                _update(col, None, k, X, Ys, w)

    def _update(self, col, Q, I, X, Ys, value):
        K = col.k
        if Ys == 0:
            # Items of the form phrase(I, X/[], K)
            item = (I, X)
            was = col.c_chart.get(item)
            if was is None:
                Q[item] = -((K - I) * self.ORDER_MAX + self.order[X])
                col.c_chart[item] = value
            else:
                col.c_chart[item] = was + value

        else:
            # Items of the form phrase(I, X/[Y|Ys], K)
            item = (I, X, Ys)
            was = col.i_chart.get(item)
            if was is None:
                col.waiting_for[self.first_Ys[Ys]].append(item)
                col.i_chart[item] = value
            else:
                col.i_chart[item] = was + value

    # We have derived the `next_token_weights` algorithm by backpropagation on
    # the program with respect to the item `phrase(0, s, K)`.
    #
    # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
    #
    # Directly applying the gradient transformation, we get
    #
    # ∇phrase(0, s/[], K) += 1
    # ∇phrase(J, Y/[], K) += phrase(I, X/[Y|Ys], J) * ∇phrase(I, X/Ys, K)
    #
    # Some quick analysis reveals that the `Ys` list must always be [], and
    # that K is always equal to the final column.  We specialize the program
    # below:
    #
    # ∇phrase(0, s/[], K) += 1
    # ∇phrase(J, Y/[], K) += phrase(I, X/[Y], J) * ∇phrase(I, X/[], K)
    #
    # We can abbreviate the names:
    #
    # q(0, s) += 1
    # q(J, Y) += phrase(I, X/[Y], J) * q(I, X)
    #
    # These items satisfy (I > J) and (X > Y) where the latter is the
    # nonterminal ordering.  Thus, we can efficiently evaluate these equations
    # by backward chaining.
    #
    # The final output is the vector
    #
    # p(W) += q(I, X) * phrase(I, X/[W], J)  where len(J) * terminal(W).
    #
    def next_token_weights(self, cols):
        is_terminal = self.cfg.is_terminal
        zero = self.cfg.R.zero

        q = {}
        q[0, self.cfg.S] = self.cfg.R.one

        col = cols[-1]
        col_waiting_for = col.waiting_for
        col_i_chart = col.i_chart

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        p = self.cfg.R.chart()

        for Y in col_waiting_for:
            if is_terminal(Y):
                total = zero
                for I, X, Ys in col_waiting_for[Y]:
                    if self.unit_Ys[Ys]:
                        node = (I, X)
                        value = self._helper(node, cols, q)
                        total += col_i_chart[I, X, Ys] * value
                p[Y] = total

        return p

    def _helper(self, top, cols, q):
        value = q.get(top)
        if value is not None:
            return value

        zero = self.cfg.R.zero
        stack = [Node(top, None, zero)]

        while stack:
            node = stack[-1]  # 👀

            # place neighbors above the node on the stack
            (J, Y) = node.node

            t = node.cursor

            if node.edges is None:
                node.edges = [x for x in cols[J].waiting_for[Y] if self.unit_Ys[x[2]]]

            # cursor is at the end, all neighbors are done
            elif t == len(node.edges):
                # clear the node from the stack
                stack.pop()
                # promote the incomplete value node.value to a complete value (q)
                q[node.node] = node.value

            else:
                (I, X, _) = arc = node.edges[t]
                neighbor = (I, X)
                neighbor_value = q.get(neighbor)
                if neighbor_value is None:
                    stack.append(Node(neighbor, None, zero))
                else:
                    # neighbor value is ready, advance the cursor, add the
                    # neighbors contribution to the nodes value
                    node.cursor += 1
                    node.value += cols[J].i_chart[arc] * neighbor_value

        return q[top]

    def generate_rust_test_case(self):
        # generates a test case in Rust code by exporting the parser state variables
        # Copy-paste the printout to `mod tests { ... }` in lib.rs to debug.

        print(
            """
    #[test]
    fn test_earley() {{

        let rhs: HashMap<u32, Vec<RHS>> = [
            {}
        ].iter().cloned().collect();
        """.format(
                ", ".join(
                    f"({x}, "
                    + "vec![{}])".format(", ".join(f"({float(u)}, {v})" for u, v in y))
                    for x, y in self.rhs.items()
                )
            )
        )

        print(
            """
        let order: HashMap<u32, u32> = [
            {}
        ].iter().cloned().collect();
        """.format(", ".join(f"({u}, {v})" for u, v in self.order.items()))
        )

        print(
            """
        let outgoing: HashMap<u32, Vec<u32>> = [
            {}
        ].iter().cloned().collect();
        """.format(
                ", ".join(
                    "({}, vec![{}])".format(i, ", ".join(map(str, s)))
                    for i, s in self.R_outgoing.items()
                )
            )
        )

        print(
            """
        let first_ys = vec![
            {}
        ].iter().cloned().collect();
        """.format(
                ", ".join(
                    f'Terminal(String::from("{y}"))'
                    if isinstance(y, str)
                    else f"Nonterminal({y})"
                    for y in self.first_Ys
                )
            )
        )

        print(
            """
        let rest_ys = vec![
            {}
        ];
        """.format(", ".join(map(str, self.rest_Ys)))
        )

        print(
            """
        let unit_ys = vec![
            {}
        ];
        """.format(", ".join(map(lambda x: str(bool(x)).lower(), self.unit_Ys)))
        )

        print(
            """
        let vocab = [
            {}
        ].iter().cloned().collect();
        """.format(", ".join(f'String::from("{v}")' for v in self.cfg.V))
        )

        print(
            """
        let empty_weight = {};
        let start = {};
        let order_max = {};
        """.format(
                sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ()),
                self.cfg.S,
                self.ORDER_MAX,
            )
        )

        print("""
        let mut earley = Earley::new(
            rhs, start, order, order_max, outgoing, first_ys,
            rest_ys, unit_ys, vocab, empty_weight,
        );
        let chart = earley.p_next(vec![]);
        dbg!(&chart);

    }}
        """)


class Node:
    __slots__ = ("value", "node", "edges", "cursor")

    def __init__(self, node, edges, value):
        self.node = node
        self.edges = edges
        self.value = value
        self.cursor = 0
