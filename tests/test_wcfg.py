import numpy as np
from arsenal import colors

from genlm_grammar.cfg import CFG, Derivation
from genlm_grammar.chart import Chart
from genlm_grammar.semiring import Boolean, Entropy, Float, Log, MaxPlus, MaxTimes, Real
from genlm_grammar.util import display_table

TOL = 1e-5


def assert_equal(have, want, tol=TOL):
    error = have.metric(want)
    assert error <= tol, f"have = {have}, want = {want}, error = {error}"


def test_sdd1():
    "This is a silly bug made once when refactoring"

    cfg = CFG.from_string(
        """
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """,
        Real,
    )

    have = cfg("aa")
    want = cfg.cnf("aa")
    assert have.metric(want) <= 1e-5, [have, want]


def test_misc():
    # Derivation(None, Derivation(None, 'X'))._repr_html_()

    x = Derivation(None, Derivation(None, "X"))
    y = Derivation(None, Derivation(None, "Y"))
    assert x == x
    assert hash(x) == hash(x)
    assert x != y
    assert hash(x) != hash(y)

    CFG.from_string("", Real)._repr_html_()

    # Derivation.to_nltk(None)

    cfg = CFG.from_string("1: X -> Y", Real)

    def f(x):
        return x.lower()

    assert cfg.rename(f).N == {f(X) for X in cfg.N}

    try:
        CFG.from_string("x -> y : 1", Real)
    except ValueError:
        pass

    display_table([[cfg, "hello"], [cfg, cfg]], headings=["a", "b"])
    display_table([[cfg, "hello"], [cfg, cfg]])

    # include an expected-failure test
    try:
        Chart(Real, {"a": Real(1)}).assert_equal({"a": Real(2)})
    except AssertionError:
        pass
    else:
        raise AssertionError("test failed")

    cfg = CFG.from_string(
        """
        1: S -> X
        1: S -> Y
        2: X -> a
        3: Y -> b
        """,
        Float,
    )
    cfg["Y"].trim().assert_equal("3: Y -> b")

    # call it twice to hit the trim cache
    cfg.trim().trim()  # serial
    cfg.trim()  # parallel

    assert (cfg @ "a").treesum() == 2
    assert (cfg @ ("a",)).treesum() == 2


def test_agenda_misc():
    # test stopping early
    g = CFG.from_string(
        """
        0.5: S → a S
        1: S → a
        """,
        Float,
    )

    g.agenda(maxiter=2).assert_equal({"a": 1, "S": 1.5})
    g.agenda(maxiter=3).assert_equal({"a": 1, "S": 1.75})


def test_semirings():
    p = Entropy.from_string("1")
    assert p.H == 0

    # uniform distrbution over 3 elements
    g = CFG.from_string(
        """

    .25: S → a
    .25: S → b
    .25: S → c
    .25: S → d
    0.0: S → e

    """,
        Entropy,
    )

    assert np.allclose(g.treesum(tol=TOL).H, 2.0)

    z = Entropy.zero
    e = Entropy.one
    x = Entropy.from_string("0.5")
    y = Entropy.from_string("0.2")
    a = Entropy.from_string("0.1")

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star() == e + x * x.star()
    assert x.star() == e + x.star() * x

    assert ((x + y) * a).metric(x * a + y * a) <= 1e-10

    g = CFG.from_string(
        """
    1: A → a
    0: B → b
    """,
        Boolean,
    )

    g.agenda().assert_equal(
        {
            "a": Boolean(True),
            "b": Boolean(True),
            "A": Boolean(True),
        }
    )

    z = Boolean.zero
    e = Boolean.one
    x = Boolean.from_string("True")
    y = Boolean.from_string("False")

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star() == e + x * x.star()
    assert x.star() == e + x.star() * x

    for a in [z, e]:
        for b in [z, e]:
            for c in [z, e]:
                assert (a + b) * c == a * c + b * c

    z = MaxPlus.zero
    e = MaxPlus.one
    x = MaxPlus.from_string("-3")
    y = MaxPlus.from_string("-4")
    w = MaxPlus.from_string("-5")

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star() == e + x * x.star()
    assert x.star() == e + x.star() * x

    for a in [w, x, y]:
        for b in [w, x, y]:
            for c in [w, x, y]:
                assert (a + b) * c == a * c + b * c

    z = MaxTimes.zero
    e = MaxTimes.one
    x = MaxTimes.from_string(".3")
    y = MaxTimes.from_string(".2")
    w = MaxTimes.from_string(".1")

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star() == e + x * x.star()
    assert x.star() == e + x.star() * x

    for a in [w, x, y]:
        for b in [w, x, y]:
            for c in [w, x, y]:
                assert (a + b) * c == a * c + b * c

    z = Log.zero
    e = Log.one
    x = Log.from_string("-3")
    y = Log.from_string("-2")
    w = Log.from_string("-1")

    assert x * e == x
    assert e * x == x
    assert x * z == z
    assert z * x == z

    assert z + x == x
    assert x + z == x

    assert x.star().metric(e + x * x.star()) <= 1e-10
    assert x.star().metric(e + x.star() * x) <= 1e-10

    for a in [w, x, y]:
        for b in [w, x, y]:
            for c in [w, x, y]:
                assert ((a + b) * c).metric(a * c + b * c) <= 1e-10

    a = MaxPlus(1)
    b = MaxPlus(2)
    assert a.metric(a) == 0
    assert a.metric(b) == 1
    assert a + b == MaxPlus(2)  # despite the name it does not add!
    assert a * b == MaxPlus(3)


def test_treesum():
    cfg = CFG.from_string(
        """
        0.25: S → S S
        0.75: S → a
        """,
        Real,
    )

    want = cfg.naive_bottom_up()
    have = cfg.agenda(tol=TOL)

    for x in want.keys() | have.keys():
        # print(x, want[x].score, have[x].score)
        assert abs(want[x].score - have[x].score) <= 0.0001 * abs(want[x].score)

    # run for fewer iterations
    have = cfg.naive_bottom_up(timeout=2)
    have = {str(k): v.score for k, v in have.items()}
    want = {"a": 1.0, "S": 0.890625}
    assert have == want, [have, want]


def test_trim():
    cfg = CFG.from_string(
        """

        0.25: S → S S
        0.75: S → a

        0.75: A → a

        1: C → D
        1: D → C

        1: B → a
        1: B → B

        """,
        Real,
    )

    have = cfg.trim()

    want = CFG.from_string(
        """

        0.25: S → S S
        0.75: S → a

        """,
        Real,
    )

    have.assert_equal(want)

    # cotrim keeps rules that build stuff bottom-up but aren't necessarily used by S.
    have = cfg.cotrim()
    want = CFG.from_string(
        """

    0.25: S → S S
    0.75: S → a

    1:    B → B
    1:    B → a
    0.75: A → a

    """,
        Real,
    )
    have.assert_equal(want)


def test_cnf():
    cfg = CFG.from_string(
        """

        1: S → S1

        1: S → A B C d

        0.5: S1 → S1

        0.1: S1 →
        0.1: A →

        1: A → a
        1: B → d
        1: C → c

        """,
        Real,
    )

    cnf = cfg.cnf
    print(cnf)

    assert not cfg.in_cnf()
    assert cnf.in_cnf()

    assert_equal(have=cnf.treesum(), want=cfg.treesum(), tol=1e-10)


def test_grammar_size_metrics():
    cfg = CFG.from_string(
        """
        1.0: S → A B C D
        0.5: S → S
        0.2: S →
        0.1: A →

        1: A → a
        1: B → d
        1: C → c
        1: D → d
        """,
        Real,
    )

    assert cfg.size == 17
    assert cfg.num_rules == 8


def test_palindrome_derivations():
    cfg = CFG.from_string(
        """
        1: S → a S a
        1: S → b S b
        1: S → c
        """,
        Real,
    )

    # s = 'a b c b a'.split()

    n = 0
    print(colors.yellow % "Derivations:")
    for t in cfg.derivations(cfg.S, 5):
        print(colors.orange % "derivation:", t)
        n += 1
    assert n == 31, n

    # W = total(cfg, s)
    # print(colors.yellow % 'total weight:', W)
    # assert W.score == 1


def test_unfold():
    cfg = CFG.from_string(
        """
        1.0: S →
        0.5: S → S a
        0.5: B → b
        """,
        Real,
    )

    new = cfg.unfold(1, 0)
    print(new)

    err = cfg.treesum(tol=TOL).metric(new.treesum(tol=TOL))
    assert err <= 1e-5, err

    new.assert_equal(
        CFG.from_string(
            """
            1.0: S →
            0.5: S → a
            0.25: S → S a a
            0.5: B → b
            """,
            Real,
        )
    )

    # unfolding terminals is not allowed
    try:
        new = cfg.unfold(1, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected error")


def test_cky():
    cfg = CFG.from_string(
        """
        1: S ->  A B
        0.1: A -> A B
        0.4: A ->
        0.5: A -> b
        0.4: B -> a
        0.5: B ->
        0.1: B -> B A
        """,
        Real,
    )

    # brute-force enumerate of the weighted language
    L = cfg.cnf.language(4)

    all_ok = True
    for x in sorted(L, key=lambda x: (-L[x].score, x))[:20]:
        have = cfg(x)
        want = L[x]
        err = have.metric(want)
        ok = err <= 1e-4
        all_ok &= ok
        if ok:
            print(colors.mark(ok), repr("⋅".join(x)), want)
        else:
            print(colors.mark(ok), repr("⋅".join(x)), colors.red % have, want)
    assert all_ok, [err, have, want]


def test_unary_cycle_removal():
    cfg = CFG.from_string(
        """
        0.5: S → A1

        0.5: A1 → B1
        0.5: B1 → C1
        0.5: C1 → A1

        0.5: C1 → C
        0.25: C1 → C1

        0.25: C1 → C0
        1.0: C0 → C

        0.5: C → c

        """,
        Float,
    )

    unaryfree = cfg.unarycycleremove(trim=False)
    assert not unaryfree.has_unary_cycle()
    unaryfree.agenda().assert_equal(cfg.agenda(), domain=cfg.N, tol=1e-10, verbose=1)


def test_truncate_length():
    cfg = CFG.from_string(
        """
        1: S → a S a
        1: S → b S b
        1: S →
        """,
        Real,
    )

    max_length = 5

    cfg_t = cfg.truncate_length(max_length)
    have = cfg_t.language(max_length * 2)

    want = cfg.materialize(max_length=max_length)

    have.assert_equal(want)
    print(have)
    assert len(have) == 7 or max_length != 5


def test_byte_conversion():
    cfg = CFG.from_string(
        """
    1.0: S -> A B
    1.0: A -> a
    2.0: B -> a 👋
    """,
        Float,
    )
    cfg_b = cfg.to_bytes()
    for x in ["a", "a👋", "👋"]:
        assert cfg_b(x.encode("utf-8")) == cfg(x), x


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
