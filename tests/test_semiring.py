from genlm.grammar.cfg import CFG
from genlm.grammar.semiring import Float, GradReal, Real
import random

toll = 1e-12

def gr():
    return GradReal(random.uniform(-10,10))

def test_associative():
    a = gr()
    b = gr()
    c = gr()

    assert ((a + b) + c).metric(a + (b + c)) < toll
    assert ((a * b) * c ).metric( a * (b * c)) < toll

def test_distributive():
    a = gr()
    b = gr()
    c = gr()

    assert (a * (b + c)).metric(a * b + a * c) < toll

def test_star():
    a = GradReal.one
    while a == GradReal.one:
        a = gr()
    r = Real(a.score)
    assert abs(a.star().score - r.star().score) < toll

