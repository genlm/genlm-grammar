from itertools import product

from arsenal import colors

import examples
from genlm.grammar.cfg import CFG
from genlm.grammar.semiring import Real


def prefix_weight_bf(self, s, depth, verbose=False):
    "Brute-force computation of the prefix weight of the sequence `s`."
    bf = self.R.zero
    for d in self.derivations(None, depth):
        y = d.Yield()
        w = d.weight()
        if len(y) >= len(s) and all(x == y for x, y in zip(s, y)):
            if verbose:
                print(d.weight(), d.Yield(), d)
            bf += w
    return bf


def test_parsing():
    cfg = CFG.from_string(
        """
        1.0: S -> A
        0.2: A -> A A
        0.4: A -> a
        0.4: A -> b

        """,
        Real,
    )

    for s in [
        "",
        "a",
        "ab",
        "abb",
        "abba",
    ]:
        print(colors.yellow % "check string:", repr(s))
        want = cfg(s)

        print("want:", want)

        # assert_equal((cfg @ s).trim().treesum(), want, 1e-5)

        print("nullable derivatives")
        other = cfg.derivatives(s)[-1].trim().null_weight_start()
        assert_equal(other, want, 1e-5)


def test_palindrome():
    cfg = CFG.from_string(
        """
        0.3: S -> a S a
        0.4: S -> b S b
        0.3: S ->
        """,
        Real,
    )

    s = "aba"

    assert_equal(
        have=prefix_weight_bf(cfg, s, 15), want=Real(0.03599999999726485), tol=1e-3
    )

    assert_equal(
        have=cfg.derivatives(s)[-1].treesum(),
        want=Real(0.03599999999726485),
    )

    s = "ababa"
    assert_equal(
        have=cfg.derivatives(s)[-1].treesum(),
        want=Real(0.004319999997413523),
    )


def test_new_palindrome():
    cfg = CFG.from_string(
        """
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """,
        Real,
    )

    s = "a"
    print(repr(s))
    assert_equal(
        have=cfg.prefix_weight(s),
        want=cfg.derivatives(s)[-1].treesum(),
    )

    s = "aa"
    print(repr(s))
    assert_equal(
        have=cfg.prefix_weight(s),
        want=cfg.derivatives(s)[-1].treesum(),
    )

    s = "ab"
    print(repr(s))
    assert_equal(
        have=cfg.prefix_weight(s),
        want=cfg.derivatives(s)[-1].treesum(),
    )

    s = "aaaba"
    print(repr(s))
    assert_equal(
        have=cfg.prefix_weight(s),
        want=cfg.derivatives(s)[-1].treesum(),
    )

    s = ""
    print(repr(s))
    assert_equal(
        have=cfg.prefix_weight(s),
        want=cfg.derivatives(s)[-1].treesum(),
    )


def test_examples():
    for name in [
        "abcd",
        "abcde_prefixes",
        "catalan",
        "catalan_ab",
        "palindrome_ab",
        "papa",
    ]:
        print("running example:", name)
        cfg = getattr(examples, name)

        for s in product(cfg.V, repeat=3):
            s = list(s)
            try:
                assert (
                    cfg.R.metric(
                        cfg.prefix_weight(s),
                        cfg.derivatives(s)[-1].treesum(),
                    )
                    < 1e-8
                )
            except AssertionError as e:
                print(name, repr(s))
                print(e)
                raise e


THROW = True


def assert_equal(have, want, tol=1e-8):
    error = have.metric(want)
    if THROW:
        assert error <= tol, f"have = {have}, want = {want}, error = {error}"
    else:
        if error <= tol:
            print(
                colors.mark(error <= tol),
                f"have = {have}, want = {want}, error = {error}",
            )
        else:
            print(
                colors.mark(error <= tol),
                f"have = {have}, want = {want}, error = {error}",
            )


def test_finite():
    cfg = CFG.from_string(
        """

    1: S -> a b c d e
    1: S -> a b c d x
    1: S -> a b c x x
    1: S -> a b x x x
    1: S -> a x x x x
    1: S -> x x x x x

    """,
        Real,
    )

    s = ""
    assert_equal(
        want=Real(6),
        have=cfg.derivatives(s)[-1].treesum(),
    )

    s = "a"
    assert_equal(
        want=Real(5),
        have=cfg.derivatives(s)[-1].treesum(),
    )

    s = "ab"
    assert_equal(
        want=Real(4),
        have=cfg.derivatives(s)[-1].treesum(),
    )

    s = "abc"
    assert_equal(
        want=Real(3),
        have=cfg.derivatives(s)[-1].treesum(),
    )

    s = "abcd"
    assert_equal(
        want=Real(2),
        have=cfg.derivatives(s)[-1].treesum(),
    )

    s = "abcde"
    assert_equal(
        want=Real(1),
        have=cfg.derivatives(s)[-1].treesum(),
    )

    s = "abcdef"
    assert_equal(
        want=Real(0),
        have=cfg.derivatives(s)[-1].treesum(),
    )


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
