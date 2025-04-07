from collections import defaultdict
from functools import cached_property
from itertools import zip_longest

from genlm.grammar.semiring import Boolean
from genlm.grammar.wfsa import EPSILON, WFSA

ε = EPSILON
ε_1 = f"{EPSILON}₁"
ε_2 = f"{EPSILON}₂"


class FST(WFSA):
    """A weighted finite-state transducer that maps between two alphabets.

    A finite-state transducer (FST) extends a weighted finite-state automaton (WFSA)
    by having two alphabets - an input alphabet A and an output alphabet B. Each transition
    is labeled with a pair (a,b) where a is from A and b is from B.

    The FST defines a weighted relation between strings over A and strings over B.
    """

    def __init__(self, R):
        """Initialize an empty FST.

        Args:
            R: The semiring for transition weights
        """
        super().__init__(R=R)

        # alphabets
        self.A = set()  # input alphabet
        self.B = set()  # output alphabet

    def add_arc(self, i, ab, j, w):  # pylint: disable=arguments-renamed
        """Add a weighted transition between states.

        Args:
            i: Source state
            ab: Tuple (a,b) of input/output symbols, or EPSILON
            j: Target state
            w: Weight of the transition

        Returns:
            self
        """
        if ab != EPSILON:
            (a, b) = ab
            self.A.add(a)
            self.B.add(b)
        return super().add_arc(i, ab, j, w)

    def set_arc(self, i, ab, j, w):  # pylint: disable=arguments-renamed
        """Set the weight of a transition between states.

        Args:
            i: Source state
            ab: Tuple (a,b) of input/output symbols, or EPSILON
            j: Target state
            w: New weight for the transition

        Returns:
            self
        """
        if ab != EPSILON:
            (a, b) = ab
            self.A.add(a)
            self.B.add(b)
        return super().set_arc(i, ab, j, w)

    def __call__(self, x, y):
        """Compute the weight of mapping input x to output y.

        If x or y is None, returns a weighted language representing the cross section.

        Args:
            x: Input string or None
            y: Output string or None

        Returns:
            Weight of mapping x to y, or a WFSA representing the cross section if x or y is None
        """
        if x is not None and y is not None:
            x = FST.from_string(x, self.R)
            y = FST.from_string(y, self.R)
            return (x @ self @ y).total_weight()

        elif x is not None and y is None:
            x = FST.from_string(x, self.R)
            return (x @ self).project(1)

        elif x is None and y is not None:
            y = FST.from_string(y, self.R)
            return (self @ y).project(0)

        else:
            return self

    @classmethod
    def from_string(cls, xs, R, w=None):
        """Create an FST that accepts only the given string with optional weight.

        Args:
            xs: Input string
            R: Semiring for weights
            w: Optional weight for the string

        Returns:
            An FST accepting only xs with weight w
        """
        return cls.diag(WFSA.from_string(xs=xs, R=R, w=w))

    @staticmethod
    def from_pairs(pairs, R):
        """Create an FST accepting the given input-output string pairs.

        Args:
            pairs: List of (input_string, output_string) tuples
            R: Semiring for weights

        Returns:
            An FST accepting the given string pairs with weight one
        """
        p = FST(R)
        p.add_I(0, R.one)
        p.add_F(1, R.one)
        for i, (xs, ys) in enumerate(pairs):
            p.add_arc(0, EPSILON, (i, 0), R.one)
            for j, (x, y) in enumerate(zip_longest(xs, ys, fillvalue=EPSILON)):
                p.add_arc((i, j), (x, y), (i, j + 1), R.one)
            p.add_arc((i, max(len(xs), len(ys))), EPSILON, 1, R.one)
        return p

    def project(self, axis):
        """Project the FST onto one of its components to create a WFSA.

        Args:
            axis: 0 for input projection, 1 for output projection

        Returns:
            A WFSA over the projected alphabet
        """
        assert axis in [0, 1]
        A = WFSA(R=self.R)
        for i, (a, b), j, w in self.arcs():
            if axis == 0:
                A.add_arc(i, a, j, w)
            else:
                A.add_arc(i, b, j, w)
        for i, w in self.I:
            A.add_I(i, w)
        for i, w in self.F:
            A.add_F(i, w)
        return A

    @cached_property
    def T(self):
        """Return the transpose of this FST by swapping input/output labels.

        Returns:
            A new FST with input/output labels swapped
        """
        T = self.spawn()
        for i, (a, b), j, w in self.arcs():
            T.add_arc(i, (b, a), j, w)  # (a,b) -> (b,a)
        for q, w in self.I:
            T.add_I(q, w)
        for q, w in self.F:
            T.add_F(q, w)
        return T

    def prune_to_alphabet(self, A, B):
        """Remove transitions with labels not in the given alphabets.

        Args:
            A: Set of allowed input symbols, or None to allow all
            B: Set of allowed output symbols, or None to allow all

        Returns:
            A new FST with invalid transitions removed
        """
        T = self.spawn()
        for i, (a, b), j, w in self.arcs():
            if (A is None or a in A) and (B is None or b in B):
                T.add_arc(i, (a, b), j, w)
        for q, w in self.I:
            T.add_I(q, w)
        for q, w in self.F:
            T.add_F(q, w)
        return T.trim

    def __matmul__(self, other):
        """Compose this FST with another FST or automaton.

        Args:
            other: Another FST, CFG or automaton to compose with

        Returns:
            The composed FST
        """
        if not isinstance(other, FST):
            from genlm.grammar.cfg import CFG

            if isinstance(other, CFG):
                return other @ self.T
            else:
                other = FST.diag(other)

        # minor efficiency trick: it's slightly more efficient to associate the composition as follows
        if len(self.states) < len(other.states):
            return (
                self._augment_epsilon_transitions(0)  # rename epsilons on the right
                ._compose(
                    epsilon_filter_fst(self.R, self.B), coarsen=False
                )  # this FST carefully combines the special epsilons
                ._compose(
                    other._augment_epsilon_transitions(1)
                )  # rename epsilons on th left
            )

        else:
            return self._augment_epsilon_transitions(
                0
            )._compose(  # rename epsilons on the right
                epsilon_filter_fst(
                    self.R, self.B
                )._compose(  # this FST carefully combines the special epsilons
                    other._augment_epsilon_transitions(1), coarsen=False
                )
            )  # rename epsilons on th left

    def _compose(self, other, coarsen=True):
        """Internal composition implementation with optional coarsening.

        Args:
            other: FST to compose with
            coarsen: Whether to apply pruning/coarsening

        Returns:
            The composed FST
        """
        if coarsen and FST.PRUNING is not None:
            keep = FST.PRUNING(self, other)  # pylint: disable=E1102
            result = self._pruned_compose(other, keep, keep.keep_arc)

        else:
            result = self._pruned_compose(
                other, lambda x: True, lambda i, label, j: True
            )

        return result

    # TODO: add assertions for the 'bad' epsilon cases to ensure users aren't using this method incorrectly.
    def _pruned_compose(self, other, keep, keep_arc):
        """Implements pruned on-the-fly composition of FSTs.

        Args:
            other: FST to compose with
            keep: Function that determines which states to keep
            keep_arc: Function that determines which arcs to keep

        Returns:
            The composed FST with pruning applied
        """
        C = FST(R=self.R)

        # index arcs in `other` to so that they are fast against later
        tmp = defaultdict(list)
        for i, (a, b), j, w in other.arcs():
            tmp[i, a].append((b, j, w))

        visited = set()
        stack = []

        # add initial states
        for P, w1 in self.I:
            for Q, w2 in other.I:
                PQ = (P, Q)

                if not keep(PQ):
                    continue

                C.add_I(PQ, w1 * w2)
                visited.add(PQ)
                stack.append(PQ)

        # traverse the machine using depth-first search
        while stack:
            P, Q = PQ = stack.pop()

            # (q,p) is simultaneously a final state in the respective machines
            if P in self.stop and Q in other.stop:
                C.add_F(PQ, self.stop[P] * other.stop[Q])
                # Note: final states are not necessarily absorbing -> fall thru

            # Arcs of the composition machine are given by a cross-product-like
            # construction that matches an arc labeled `a:b` with an arc labeled
            # `b:c` in the left and right machines respectively.
            for (a, b), Pʼ, w1 in self.arcs(P):
                for c, Qʼ, w2 in tmp[Q, b]:
                    assert b != EPSILON

                    PʼQʼ = (Pʼ, Qʼ)

                    if not keep(PʼQʼ) or not keep_arc(PQ, (a, c), PʼQʼ):
                        continue

                    C.add_arc(PQ, (a, c), PʼQʼ, w1 * w2)

                    if PʼQʼ not in visited:
                        stack.append(PʼQʼ)
                        visited.add(PʼQʼ)

        return C

    def _augment_epsilon_transitions(self, idx):
        """Augments the FST by changing the appropriate epsilon transitions to
        epsilon_1 or epsilon_2 transitions to be able to perform the composition
        correctly.  See Fig. 7 on p. 17 of Mohri, "Weighted Automata Algorithms".

        Args:
            idx: 0 if this is the first FST in composition, 1 if second

        Returns:
            FST with augmented epsilon transitions
        """
        assert idx in [0, 1]

        T = self.spawn(keep_init=True, keep_stop=True)

        for i in self.states:
            if idx == 0:
                T.add_arc(i, (ε, ε_1), i, self.R.one)
            else:
                T.add_arc(i, (ε_2, ε), i, self.R.one)
            for ab, j, w in self.arcs(i):
                if idx == 0 and ab[1] == ε:
                    ab = (ab[0], ε_2)
                elif idx == 1 and ab[0] == ε:
                    ab = (ε_1, ab[1])
                T.add_arc(i, ab, j, w)

        return T

    @classmethod
    def diag(cls, fsa):
        """Convert FSA to diagonal FST that maps strings to themselves.

        Args:
            fsa: Input FSA to convert

        Returns:
            FST that maps each string accepted by fsa to itself with same weight
        """
        fst = cls(fsa.R)
        for i, a, j, w in fsa.arcs():
            fst.add_arc(i, (a, a), j, w)
        for i, w in fsa.I:
            fst.add_I(i, w)
        for i, w in fsa.F:
            fst.add_F(i, w)
        return fst

    def coarsen(self, N, A, B):
        """Create coarsened Boolean FST by mapping states and symbols.

        Args:
            N: Function mapping states to coarsened states
            A: Function mapping input symbols to coarsened input symbols
            B: Function mapping output symbols to coarsened output symbols

        Returns:
            Coarsened Boolean FST
        """
        m = FST(Boolean)
        for i in self.start:
            m.add_I(N(i), Boolean.one)
        for i in self.stop:
            m.add_F(N(i), Boolean.one)
        for i, (a, b), j, _ in self.arcs():
            m.add_arc(N(i), (A(a), B(b)), N(j), Boolean.one)
        return m


def epsilon_filter_fst(R, Sigma):
    """Create epsilon filter FST for composition.

    Creates a 3-state FST that handles epsilon transitions correctly during
    composition by filtering invalid epsilon paths.

    Args:
        R: Semiring for weights
        Sigma: Alphabet of non-epsilon symbols

    Returns:
        Epsilon filter FST
    """
    F = FST(R)

    F.add_I(0, R.one)

    for a in Sigma:
        F.add_arc(0, (a, a), 0, R.one)
        F.add_arc(1, (a, a), 0, R.one)
        F.add_arc(2, (a, a), 0, R.one)

    F.add_arc(0, (ε_2, ε_1), 0, R.one)
    F.add_arc(0, (ε_1, ε_1), 1, R.one)
    F.add_arc(0, (ε_2, ε_2), 2, R.one)

    F.add_arc(1, (ε_1, ε_1), 1, R.one)
    F.add_arc(2, (ε_2, ε_2), 2, R.one)

    F.add_F(0, R.one)
    F.add_F(1, R.one)
    F.add_F(2, R.one)

    return F


FST.PRUNING = None
