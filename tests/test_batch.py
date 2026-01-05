import numpy as np
from arsenal import colors

from genlm.grammar.cfg import CFG, Derivation
from genlm.grammar.chart import Chart
from genlm.grammar.semiring import Boolean, Entropy, Float, Log, MaxPlus, MaxTimes, Real, GradReal
from genlm.grammar.util import display_table
from genlm.grammar.parse.earley import Earley


def test_nullary_fast():
    from examples import palindrome_ab

    batch_grammar = palindrome_ab.batch_grammar
    nullary_standard = batch_grammar.nullaryremove()
    nullary_fast = batch_grammar.nullaryremove_special()

    assert not nullary_fast.has_nullary()

    for string in ["","ab","aabb","aaaa"]:
        assert abs( nullary_standard(string) -nullary_fast(string)) < 1e-10

def test_next_token_palindrome_ab():

   from examples import palindrome_ab

   pg_ab = palindrome_ab.binarize().prefix_grammar

   for string in ["","ab","aabb","aaba"]:
        for a in palindrome_ab.V:
            want = pg_ab(string+a)
            have = palindrome_ab.next_token(string)[a]
            assert abs(have - want) < 1e-7, f"string: {string}, a: {a}, have: {have}, want: {want}"

def test_next_token_catalan_ab():
    from examples import catalan_ab

    pg_ab = catalan_ab.binarize().prefix_grammar

    for string in ["","ab","aabb","aaba"]:
        for a in catalan_ab.V:
            want = pg_ab(string+a)
            have = catalan_ab.next_token(string)[a]
            assert abs(have - want) < 1e-7, f"string: {string}, a: {a}, have: {have}, want: {want}"
