"""
Implementation of weighted finite-state automata over fields.

This implementation of equivalence and minimization follows Kiefer (2020) very closely


References

 - Stefan Kiefer (2020) "Notes on Equivalence and Minimization of Weighted Automata"
   arXiv:2009.01217v1 [cs.FL]


"""

from collections import deque
from functools import cached_property

import numpy as np
from numpy import linalg

from genlm.grammar.semiring import Float
from genlm.grammar.wfsa import base

# from scipy import linalg


EPSILON = base.EPSILON


class WFSA(base.WFSA):
    """
    Weighted finite-state automata where weights are a field (e.g., real-valued).
    """

    def __init__(self, R=Float):
        super().__init__(R=R)

    def __hash__(self):
        return hash(self.simple)

    def threshold(self, threshold):
        "Drop init, arcs, final below a given abs-threshold."
        m = self.__class__(self.R)
        for q, w in self.I:
            if abs(w) >= threshold:
                m.add_I(q, w)
        for i, a, j, w in self.arcs():
            if abs(w) >= threshold:
                m.add_arc(i, a, j, w)
        for q, w in self.F:
            if abs(w) >= threshold:
                m.add_F(q, w)
        return m

    def graphviz(
        self,
        fmt=lambda x: f"{round(x, 3):g}" if isinstance(x, (float, int)) else str(x),
        **kwargs,
    ):  # pylint: disable=arguments-differ
        return super().graphviz(fmt=fmt, **kwargs)

    @cached_property
    def simple(self):
        self = self.epsremove.renumber

        S = self.dim
        start = np.full(S, self.R.zero)
        arcs = {a: np.full((S, S), self.R.zero) for a in self.alphabet}
        stop = np.full(S, self.R.zero)

        for i, w in self.I:
            start[i] += w
        for i, a, j, w in self.arcs():
            arcs[a][i, j] += w
        for i, w in self.F:
            stop[i] += w

        assert EPSILON not in arcs

        return Simple(start, arcs, stop)

    def __eq__(self, other):
        return self.simple == other.simple

    def counterexample(self, other):
        return self.simple.counterexample(other.simple)

    @cached_property
    def min(self):
        return self.simple.min.to_wfsa()

    #    @cached_property
    #    def epsremove(self):
    #        return self.simple.to_wfsa()

    def multiplicity(self, m):
        return WFSA.lift(EPSILON, m) * self

    @classmethod
    def lift(cls, x, w, R=None):
        if R is None:
            R = Float
        m = cls(R=R)
        m.add_I(0, R.one)
        m.add_arc(0, x, 1, w)
        m.add_F(1, R.one)
        return m


#    @property
#    def zero(self):
#        return self.__class__(self.R)

#    @property
#    def one(self):
#        return self.__class__.lift(EPSILON, self.R.one)

WFSA.zero = WFSA()
WFSA.one = WFSA.lift(EPSILON, w=Float.one, R=Float)


class Simple:
    def __init__(self, start, arcs, stop):
        assert EPSILON not in arcs
        self.start = start
        self.arcs = arcs
        self.stop = stop
        [self.dim] = start.shape

    #    def __call__(self, xs):
    #        forward = self.start.copy()
    #        for x in xs:
    #            if x not in self.arcs: return 0
    #            forward = forward @ self.arcs[x]
    #        return forward @ self.stop

    def __eq__(self, other):
        #        return (self is other) or hash(self) == hash(other) and self.counterexample(other) is None
        return self.counterexample(other) is None

    def __hash__(self):
        return 0

    def counterexample(self, B):
        alphabet = set(self.arcs) | set(B.arcs)

        va = self.start @ self.stop
        vb = B.start @ B.stop
        if not approx_equal(va, vb):
            return ((), va, vb)

        eta = np.hstack([self.stop, B.stop])
        if approx_equal(eta, 0):
            return  # both empty; thus, equivalent.

        worklist = deque()
        worklist.append(((), self.stop, B.stop))
        basis = [eta]

        while worklist:
            (w, VA, VB) = worklist.pop()

            for a in alphabet:
                ua = self.arcs[a] @ VA if a in self.arcs else 0 * VA
                ub = B.arcs[a] @ VB if a in B.arcs else 0 * VB

                w1 = (a, w)

                # counterexample?
                va = self.start @ ua
                vb = B.start @ ub
                if not approx_equal(va, vb):
                    return (w1, va, vb)  # yup!

                u = np.hstack([ua, ub])
                q = proj(u, basis)

                # add to the basis if it's not redundant; updates to the basis
                # go on the worklist.
                if not approx_equal(u - q, u):
                    worklist.append((w1, ua, ub))
                    basis.append(q)

    @cached_property
    def min(self):
        # Proposition 3.4: Forward (backward) conjugates are forward (backward) minimal.
        # Proposition 3.5: The backward (forward) conjugate of a forward (backward) minimal automaton is minimal.
        return self.forward_conjugate().backward_conjugate()

    def __repr__(self):
        return f"<Simple states={len(self.start)}, syms={len(self.arcs)}>"

    def forward_conjugate(self):
        # The forward basis F is an n' x n where n is the set of old states, and n' the set of new states
        #
        # we want to solve
        #
        #   old_start = new_start    F
        #       n          n'     (n' x n)
        #
        # There are n >= n' constraints, but they aren't linearly independent so
        # the system has a unique solution.
        #
        #    old_start @ F.T = new_start @ F @ F.T
        #                                 (n' x n) (n x n')
        #        n
        #
        #    old_start @ F.T @ inv(F @ F.T) = new_start
        F = self.forward_basis()

        # P = F.T @ linalg.inv(F @ F.T)
        P = linalg.pinv(F)

        # start = self.start @ P
        # assert np.allclose(self.start, start @ F)

        # We also need to solve for our new transition function
        #                   F M = M' F
        #               F M F.T = M' (F F.T)
        #    F M F.T inv(F F.T) = M' (F F.T) inv(F F.T)
        return Simple(
            start=self.start @ P,
            arcs={a: F @ M @ P for a, M in self.arcs.items()},  # apply change of basis
            stop=F @ self.stop,
        )

    @cached_property
    def reverse(self):
        return Simple(
            start=self.stop,
            arcs={a: M.T for a, M in self.arcs.items()},
            stop=self.start,
        )

    def forward_basis(self):
        worklist = [self.start]
        basis = [self.start]
        while worklist:
            V = worklist.pop()
            for a in self.arcs:
                u = V @ self.arcs[a]
                q = proj(u, basis)
                if not approx_equal(u - q, u):
                    worklist.append(u)
                    basis.append(q)
        return np.array(basis)

    def backward_conjugate(self):
        return self.reverse.forward_conjugate().reverse

    def to_wfsa(self):
        m = WFSA()
        for i in range(self.dim):
            m.add_I(i, self.start[i])
        for a in self.arcs:
            for i in range(self.dim):
                for j in range(self.dim):
                    m.add_arc(i, a, j, self.arcs[a][i, j])
        for i in range(self.dim):
            m.add_F(i, self.stop[i])
        return m


def approx_equal(x, y):
    return np.allclose(x, y)


def proj(u, Q):
    # simple implementation of the Gram–Schmidt process
    # https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    for q in Q:
        u = u - ((q @ u) / (q @ q)) * q
    return u
