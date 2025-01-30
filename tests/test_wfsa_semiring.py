from genlm_grammar.semiring import Semiring
from genlm_grammar.wfsa.base import WFSA

# def test_basics():
#    from semirings import Symbol
#
#    a = WFSA.lift('a', Real(3))
#    b = WFSA.lift('b', Real(5))
#    c = a * b
#
#    assert a('a') == Real(3)
#    assert b('b') == Real(5)
#    assert (a * b)('ab') == Real(15)
#
#    assert (a + b)('a') == Real(3)
#    assert (a + b)('b') == Real(5)
#
#    a = WFSA.lift('a', Symbol('a') + Symbol('A').star())
#    b = WFSA.lift('b', Symbol('b'))
#    c = (a * b + b.star() * a).star()
#
#    print(c('ab'))


def test_det():
    precision = 5

    # TODO: this is a quantized real class (see hash and equality methods)
    class Real(Semiring):
        #        def __init__(self, score):
        #            self.score = score

        def __eq__(self, other):
            #            return isinstance(other, Real) and self.score == other.score
            return isinstance(other, Real) and round(self.score, precision) == round(
                other.score, precision
            )

        def __hash__(self):
            #            return hash(self.score)
            return hash(round(self.score, precision))

        def __pow__(self, k):
            # unlike base impl, this ne supports arbitrary values for k (incl negative and non-integer)
            return Real(self.score**k)

        def __add__(self, other):
            return Real(self.score + other.score)

        def __mul__(self, other):
            return Real(self.score * other.score)

        def star(self):
            return Real(1 / (1 - self.score))

        def __repr__(self):
            return repr(self.score)

    Real.zero = Real(0)
    Real.one = Real(1)

    a = WFSA.lift("a", Real(3))
    b = WFSA.lift("b", Real(5))
    c = WFSA.lift("c", Real(7))

    assert (a * b).determinize("ab") == Real(15)
    assert (a * b).determinize("") == Real(0)

    assert (a * b + a * c)("ab") == Real(15)
    assert (a * b + a * c).push("ab") == Real(15)
    assert (a * b + a * c).epsremove("ab") == Real(15)

    D = (a * b + a * c).determinize
    print(D)
    assert D("ab") == Real(15)

    M = (a * b + a * c).min_det
    print(M)
    assert M("ab") == Real(15)
    print(M)


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
