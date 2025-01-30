from genlm_grammar.cfg import CFG
from genlm_grammar.semiring import Float

tol = 1e-8


def test_1():
    """The expected lenght of a string in cfg_1 is E[L] =  Σ_{n=0}^{\infty} (n+1)*(0.7^n * 0.3) = 10/3
    Hint: differentiate the geometric series !"""

    cfg_1 = CFG.from_string(
        """
        0.7: S → a S
        0.3: S → a
        """,
        Float,
    )

    want = 10 / 3
    have = cfg_1.expected_length

    assert abs(want - have) < tol


def test_2():
    """The expected lenght of a string in cfg_2 is E[L] =  Σ_{n=0}^{\infty} 2n*(0.9^n * 0.1) = 18
    Hint: differentiate the geometric series !"""

    cfg_2 = CFG.from_string(
        """
        0.9: S → a S b
        0.1: S →
        """,
        Float,
    )

    want = 18
    have = cfg_2.expected_length

    assert abs(want - have) < tol


def test_finite():
    cfg_finite = CFG.from_string(
        """
        0.5: S → a a a
        0.5: S → a
        """,
        Float,
    )

    want = 2
    have = cfg_finite.expected_length

    assert abs(want - have) < tol


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
