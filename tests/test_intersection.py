from itertools import product

from genlm_grammar.cfg import CFG, Other
from genlm_grammar.fst import FST
from genlm_grammar.semiring import Float, Real
from genlm_grammar.wfsa import EPSILON, WFSA


def assert_equal(have, want, tol=1e-5):
    if isinstance(have, (int, float)):
        error = abs(have - want)
    else:
        error = have.metric(want)
    assert error <= tol, f"have = {have}, want = {want}, error = {error}"


# reference implementation of the intersection algorithm
def intersect_slow(self, fst):
    # coerce something sequence like into a diagonal FST
    if isinstance(fst, (str, tuple)):
        fst = FST.from_string(fst, self.R)

    # coerce something FSA-like into an FST, might throw an error
    if not isinstance(fst, FST):
        fst = FST.diag(fst)

    return compose_naive_epsilon(self, fst)


def compose_naive_epsilon(self, fst):
    "Reference implementation of the grammar-transducer composition."

    # coerce something sequence like into a diagonal FST
    if isinstance(fst, (str, tuple)):
        fst = FST.from_string(fst, self.R)

    # coerce something FSA-like into an FST, might throw an error
    if not isinstance(fst, FST):
        fst = FST.diag(fst)

    new_start = self.S
    new = self.spawn(S=new_start)

    for r in self:
        for qs in product(fst.states, repeat=1 + len(r.body)):
            new.add(
                r.w,
                (qs[0], r.head, qs[-1]),
                *((qs[i], r.body[i], qs[i + 1]) for i in range(len(r.body))),
            )

    for qi, wi in fst.start.items():
        for qf, wf in fst.stop.items():
            new.add(wi * wf, new_start, (qi, Other(self.S), qf))

    for i, (a, b), j, w in fst.arcs():
        if b == EPSILON:
            new.add(w, (i, a, j))
        else:
            new.add(w, (i, a, j), b)

    for qs in product(fst.states, repeat=3):
        for a in self.V:
            new.add(
                self.R.one,
                (qs[0], a, qs[2]),
                (qs[0], EPSILON, qs[1]),
                (qs[1], a, qs[2]),
            )

    for qs in product(fst.states, repeat=3):
        new.add(
            self.R.one,
            (qs[0], Other(self.S), qs[2]),
            (qs[0], Other(self.S), qs[1]),
            (qs[1], EPSILON, qs[2]),
        )

    for qs in product(fst.states, repeat=2):
        new.add(self.R.one, (qs[0], Other(self.S), qs[1]), (qs[0], self.S, qs[1]))

    return new


def check_fst(cfg, fst):
    want = compose_naive_epsilon(cfg, fst).trim(bottomup_only=True)
    have = cfg @ fst  # fast composition

    if 0:
        want = want.trim().trim()
        have = have.trim().trim()

    print()
    print("have=")
    print(have)
    print()
    print("want=")
    print(want)

    print()
    print("have chart=")
    print(have.agenda())
    print()
    print("want chart=")
    print(want.agenda())

    assert_equal(have.treesum(), want.treesum())

    print()
    print(have)
    print()
    print(want)
    have.assert_equal(want, verbose=True)


def test_palindrome1():
    cfg = CFG.from_string(
        """
        0.3: S -> a S a
        0.4: S -> b S b
        0.3: S ->
        """,
        Float,
    )

    fsa = WFSA.from_string("aa", cfg.R)

    check(cfg, fsa)


def test_palindrome2():
    cfg = CFG.from_string(
        """
        0.3: S -> a S a
        0.4: S -> b S b
        0.3: S ->
        """,
        Real,
    )

    fsa = WFSA(Real)
    fsa.add_arc(0, "a", 0, Real.one)
    fsa.add_arc(0, "b", 0, Real.one)
    fsa.add_arc(0, "c", 0, Real.one)

    fsa.add_I(0, Real.one)
    fsa.add_F(0, Real.one)

    check(cfg, fsa)


def test_palindrome3():
    cfg = CFG.from_string(
        """
        0.3: S -> a S a
        0.4: S -> b S b
        0.3: S ->
        """,
        Real,
    )

    fsa = WFSA(Real)
    # straight line aaa
    fsa.add_arc(0, "a", 1, Real.one)
    fsa.add_arc(1, "a", 2, Real.one)
    fsa.add_arc(2, "a", 3, Real.one)
    # and then a cycle
    fsa.add_arc(3, "a", 3, Real(0.5))
    fsa.add_arc(3, "b", 3, Real(0.5))

    fsa.add_I(0, Real.one)
    fsa.add_F(3, Real.one)

    check(cfg, fsa)


def test_catalan1():
    cfg = CFG.from_string(
        """
        0.4: S -> S S
        0.3: S -> a
        0.3: S -> b
        """,
        Real,
    )

    #    fsa = WFSA.from_string('aa', cfg.R)

    check(cfg, "aa")


def test_catalan2():
    cfg = CFG.from_string(
        """
        0.4: S -> S S
        0.3: S -> a
        0.3: S -> b
        """,
        Real,
    )

    fsa = WFSA(Real)
    fsa.add_I(0, Real.one)
    fsa.add_F(3, Real.one)

    # straight line aaa
    fsa.add_arc(0, "a", 1, Real.one)
    fsa.add_arc(1, "a", 2, Real.one)
    fsa.add_arc(2, "a", 3, Real.one)
    # and then a cycle
    fsa.add_arc(3, "a", 3, Real(0.5))
    fsa.add_arc(3, "b", 3, Real(0.5))

    check(cfg, fsa)


def check(cfg, fsa):
    want = intersect_slow(cfg, fsa).trim(bottomup_only=True)
    have = cfg @ fsa  # fast intersection

    want = want.trim().trim()
    have = have.trim().trim()

    print()
    print("have=")
    print(have)
    print()
    print("want=")
    print(want)

    print()
    print("have chart=")
    print(have.agenda())
    print()
    print("want chart=")
    print(want.agenda())

    assert_equal(have.treesum(), want.treesum())

    print()
    print(have)
    print()
    print(want)
    have.assert_equal(want, verbose=True)


# COMPOSITION TESTS
def test_catalan_fst():
    cfg = CFG.from_string(
        """
        0.4: S -> S S
        0.3: S -> a
        0.3: S -> b
        """,
        Real,
    )

    fst = FST(Real)

    fst.add_I(0, Real(1.0))
    fst.add_arc(0, ("a", "b"), 1, Real(1.0))
    fst.add_arc(1, ("a", "b"), 2, Real(1.0))
    fst.add_arc(2, ("a", EPSILON), 3, Real(1.0))
    fst.add_arc(3, ("a", "b"), 3, Real(1.0))
    fst.add_arc(3, ("b", "a"), 3, Real(1.0))
    fst.add_F(3, Real(1.0))

    check_fst(cfg, fst)


def test_palindrome_fst():
    cfg = CFG.from_string(
        """
        0.3: S -> a S a
        0.4: S -> b S b
        0.3: S ->
        """,
        Real,
    )

    fst = FST(Real)

    fst.add_I(0, Real(1.0))
    fst.add_arc(0, ("a", "b"), 1, Real(1.0))
    fst.add_arc(1, ("a", "b"), 2, Real(1.0))
    fst.add_arc(2, ("a", "b"), 3, Real(1.0))
    fst.add_arc(3, ("a", EPSILON), 3, Real(1.0))
    fst.add_arc(3, ("b", EPSILON), 3, Real(1.0))
    fst.add_F(3, Real(1.0))

    check_fst(cfg, fst)


# TEST FOR COMPOSITION WITH EPSILON INPUT ARCS


def test_epsilon_fst():
    cfg = CFG.from_string(
        """
        0.3: S -> a S a
        0.4: S -> b S b
        0.3: S ->
        """,
        Real,
    )

    fst = FST(Real)

    fst.add_I(0, Real(1.0))
    fst.add_arc(0, ("a", "a"), 1, Real(1.0))
    fst.add_arc(1, (EPSILON, "a"), 2, Real(1.0))
    fst.add_arc(2, ("a", "a"), 3, Real(1.0))
    fst.add_arc(3, (EPSILON, "b"), 4, Real(1.0))
    fst.add_F(4, Real(1.0))

    fst_removed = FST(Real)

    fst_removed.add_I(0, Real(1.0))
    fst_removed.add_arc(0, ("a", "a"), 1, Real(1.0))
    fst_removed.add_arc(1, ("a", "a"), 2, Real(1.0))
    fst_removed.add_F(2, Real(1.0))

    want = compose_naive_epsilon(cfg, fst)
    have = cfg @ fst_removed

    assert_equal(want.treesum(), have.treesum())

    # check that the output of the fast implementation is a trim
    have.assert_equal(have.trim(bottomup_only=True))


def test_epsilon_fst_2():
    # This test case is a bit more complex as it contains epsilon cycles on the FST
    cfg = CFG.from_string(
        """
        0.3: S -> a S a
        0.4: S -> b S b
        0.3: S ->
        """,
        Real,
    )

    fst = FST(Real)

    fst.add_I(0, Real(1.0))
    fst.add_arc(0, ("a", "a"), 1, Real(1.0))
    fst.add_arc(1, (EPSILON, EPSILON), 1, Real(0.5))
    fst.add_arc(1, ("a", "a"), 2, Real(1.0))
    fst.add_F(2, Real(1.0))

    fst_removed = FST(Real)

    fst_removed.add_I(0, Real(1.0))
    fst_removed.add_arc(
        0, ("a", "a"), 1, Real(2.0)
    )  # The weight of the cycle has been pushed here
    fst_removed.add_arc(1, ("a", "a"), 2, Real(1.0))
    fst_removed.add_F(2, Real(1.0))

    have = cfg @ fst_removed
    want = compose_naive_epsilon(cfg, fst)

    assert_equal(want.treesum(), have.treesum())

    # check that th eoutput of teh fast implementation is a trim
    have.assert_equal(have.trim(bottomup_only=True))


def test_simple_epsilon():
    g = CFG.from_string(
        """
        1: S -> a
        """,
        Float,
    )

    t = FST(Float)
    t.add_I(0, 1)
    t.add_arc(0, ("a", ""), 1, 1)
    t.add_arc(1, ("", "b"), 1, 0.5)
    t.add_F(1, 1)

    gt = g @ t

    assert_equal(gt(""), 1.0)
    assert_equal(gt("b"), 0.5)
    assert_equal(gt("bb"), 0.25)
    assert_equal(gt("bbb"), 0.125)
    assert_equal(gt("bbbb"), 0.0625)
    assert_equal(gt("bbbbb"), 0.03125)


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
