from genlm_grammar import CFG, Float, EPSILON, FST


def test_fst_cfg1():
    fst = FST(Float)

    fst.add_I(0, 1.0)
    fst.add_arc(0, ("a", "A"), 0, 0.5)
    fst.add_arc(0, ("b", "B"), 0, 0.5)
    fst.add_arc(0, ("c", "C"), 0, 0.5)
    fst.add_F(0, 1.0)

    # apply from the left of the transducer
    have = (
        CFG.from_string(
            """
            1: S -> a b c
            """,
            Float,
        )
        @ fst
    )

    have = have.trim()

    assert have.V == {"A", "B", "C"}

    have.language(5).assert_equal(Float.chart({("A", "B", "C"): 0.125}), tol=1e-10)

    # apply from the right of the transducer
    have = fst @ CFG.from_string(
        """
        1: S -> A B C
        """,
        Float,
        is_terminal=lambda X: X in "ABC",
    )

    assert have.V == {"a", "b", "c"}

    have.language(5).assert_equal(Float.chart({("a", "b", "c"): 0.125}), tol=1e-10)


def test_basic1():
    fst = FST(Float)

    fst.add_I(0, 1.0)
    fst.add_arc(0, ("a", "b"), 0, 0.5)
    fst.add_F(0, 1.0)

    assert fst("aaa", "bbb") == 0.5**3
    assert fst("", "") == 1
    assert fst("a", "a") == 0

    # TODO: none of these are working yet
    x = fst("a", None)

    assert x("a") == 0  # should equal 0
    assert x("b") == 0.5  # should equal .5
    assert x("") == 0

    x = fst(None, "b")

    assert x("a") == 0.5  # should equal .5
    assert x("b") == 0  # should equal 0
    assert x("") == 0


def test_basic2():
    a2b = FST(Float)
    a2b.add_I(0, 1.0)
    a2b.add_arc(0, ("a", "b"), 0, 0.5)
    a2b.add_F(0, 1.0)

    assert a2b("aaa", "bbb") == 0.5**3

    b2c = FST(Float)
    b2c.add_I(0, 1.0)
    b2c.add_arc(0, ("b", "c"), 0, 0.5)
    b2c.add_F(0, 1.0)

    assert b2c("bb", "cc") == 0.5**2

    a2c = a2b @ b2c
    assert a2c("aaa", "ccc") == (0.5 * 0.5) ** 3


def test_prefixes():
    m = FST(Float)

    alphabet = "abcd"

    m.add_I(0, 1.0)
    m.add_F(1, 1.0)
    for a in alphabet:
        m.add_arc(0, (a, a), 0, 1.0)

    # transition
    m.add_arc(0, (EPSILON, EPSILON), 1, 1.0)

    for a in alphabet:
        m.add_arc(1, (a, EPSILON), 1, 1.0)

    print(m("abcd", None).renumber.epsremove.trim)

    # a length 4 string has 5 prefixes (thanks to epsilon)
    assert m("abcd", None).total_weight() == 5


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
