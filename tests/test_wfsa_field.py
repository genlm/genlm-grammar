import numpy as np
import pytest
from arsenal import colors

from genlm.grammar.wfsa.field_wfsa import WFSA


def test_misc():
    one = WFSA.one
    zero = WFSA.zero

    a = WFSA.lift("a", 1)
    b = WFSA.lift("b", 1)
    c = WFSA.lift("c", 1)

    M = zero + one + a + a * b + a * b * c

    # dry run
    M.graphviz()

    with pytest.warns(UserWarning, match="empty visualization"):
        zero.graphviz()

    M._repr_svg_()


def test_min():
    one = WFSA.one
    zero = WFSA.zero

    a = WFSA.lift("a", 1)
    b = WFSA.lift("b", 1)
    c = WFSA.lift("c", 1)

    print(colors.yellow % "Example 1")
    M = one + a + a * b + a * b * c
    m = M.min
    assert m.dim == 4, m.dim

    print(colors.yellow % "Example 2")
    z = zero
    for x1 in [a, b]:
        for x2 in [a, b]:
            for x3 in [a, b]:
                z += x1 * x2 * x3
    assert z.min.dim == 4

    print(colors.yellow % "Example 3")
    assert ((a + b) * (a + b) * (a + b)).min.dim == 4

    print(colors.yellow % "Example 4")
    M = (one + a * a.star()) * b
    m = M.min
    # print(m)
    # print(m.to_wfsa().push())
    assert m.dim == 2, m.dim

    M = WFSA.one + a + a * b + a * b * c
    m = M.min
    C = compare_language(M.min, M, M.alphabet, 5)
    assert C.max_err <= 1e-10

    # M = (a + b + c) * (a + b + c)
    M = (a * a + a * b + a * c) + (b * a + b * b + b * c) + (c * a + c * b + c * c)
    m = M.min
    C = compare_language(M.min, M, M.alphabet, 5)
    assert C.max_err <= 1e-10
    assert m.dim == 3

    M = (a + b + c) * (a + b + c)
    m = M.min
    C = compare_language(M.min, M, M.alphabet, 5)
    assert C.max_err <= 1e-10
    assert m.dim == 3


def compare_language(have, want, alphabet, length):
    from itertools import product

    from arsenal.maths import compare

    A = {}
    B = {}
    for t in range(length + 1):
        for x in product(alphabet, repeat=t):
            B[x] = have(x)
            A[x] = want(x)
    return compare(A, B, verbose=0)


def test_equivalence():
    a = WFSA.lift("a", 1)
    b = WFSA.lift("b", 1)
    c = WFSA.lift("c", 1)

    one = WFSA.one

    # ___________________________________________________________________________
    #
    print(colors.yellow % "Example 0")
    A = a.star()
    B = a.star()

    w = A.counterexample(B)
    # print('counterexample?', w)
    assert w is None

    assert np.allclose(A("a"), 1)
    assert np.allclose(A("a"), 1)

    assert np.allclose(A("b"), 0)
    assert np.allclose(B("b"), 0)

    # ___________________________________________________________________________
    #
    print(colors.yellow % "Example 1")
    A = a * (b + c)
    B = a * b + a * c

    w = A.counterexample(B)
    # print('counterexample?', w)
    assert w is None

    assert np.allclose(A("ab"), 1), A("ab")
    assert np.allclose(A("ac"), 1)
    assert np.allclose(A("ab"), 1)
    assert np.allclose(B("ac"), 1)

    # ___________________________________________________________________________
    #
    print(colors.yellow % "Example 2")
    A = a * (b + c + c)
    B = a * b + a * c

    w = A.counterexample(B)
    # print('counterexample?', w)
    assert w is not None

    # ___________________________________________________________________________
    #
    print(colors.yellow % "Example 3")
    A = a * (b + c + c)
    B = a * b + a * c + a * c

    w = A.counterexample(B)
    # print('counterexample?', w)
    assert w is None

    # ___________________________________________________________________________
    #
    print(colors.yellow % "Example 4")
    A = a * b.star() * c.star()
    B = a * (one + b * b.star()) * c.star()

    w = A.counterexample(B)
    # print('counterexample?', w)
    assert w is None

    # ___________________________________________________________________________
    #
    print(colors.yellow % "Example 5")
    A = a.star()
    B = one + a * a.star()

    w = A.counterexample(B)
    # print('counterexample?', w)
    assert w is None

    # ___________________________________________________________________________
    #
    print(colors.yellow % "Example 6")

    A = (one + a * a.star()) * b
    B = a.star() * b
    assert A.counterexample(B) is None

    w = A.counterexample(B)
    # print('counterexample?', w)
    assert w is None

    # ___________________________________________________________________________
    #
    print(colors.yellow % "Example 7")
    A = (a * b).star()
    B = one + a * (b * a).star() * b

    w = A.counterexample(B)
    # print('counterexample?', w)
    assert w is None

    # ___________________________________________________________________________
    #
    print(colors.yellow % "Example 8")
    a = WFSA.lift("a", 0.1)
    b = WFSA.lift("b", 0.2)
    c = WFSA.lift("c", 0.3)

    A = (a + b).star()
    B = (a.star() * b).star() * a.star()

    w = A.counterexample(B)
    # print('counterexample?', w)
    assert w is None


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
