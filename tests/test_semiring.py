from genlm.grammar.semiring import Log
import random


log_toll = 1e-9


def lg():
    return Log(random.uniform(-10, 10))


def test_log_associative():
    a = lg()
    b = lg()
    c = lg()

    assert ((a + b) + c).metric(a + (b + c)) < log_toll
    assert ((a * b) * c).metric(a * (b * c)) < log_toll


def test_log_commutative():
    a = lg()
    b = lg()

    assert (a + b).metric(b + a) < log_toll
    assert (a * b).metric(b * a) < log_toll


def test_log_identities():
    a = lg()

    assert (a + Log.zero).metric(a) < log_toll
    assert (Log.zero + a).metric(a) < log_toll
    assert (a * Log.one).metric(a) < log_toll
    assert (Log.one * a).metric(a) < log_toll


def test_log_annihilation():
    a = lg()

    assert (a * Log.zero) == Log.zero
    assert (Log.zero * a) == Log.zero


def test_log_distributive():
    a = lg()
    b = lg()
    c = lg()

    assert (a * (b + c)).metric(a * b + a * c) < log_toll
    assert ((b + c) * a).metric(b * a + c * a) < log_toll


def test_log_star():
    a = Log(random.uniform(-10, -0.01))
    assert a.star().metric(Log.one + (a * a.star())) < log_toll
