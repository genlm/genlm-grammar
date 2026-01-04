import numpy as np
from arsenal import colors

from genlm.grammar.cfg import CFG, Derivation
from genlm.grammar.chart import Chart
from genlm.grammar.semiring import Boolean, Entropy, Float, Log, MaxPlus, MaxTimes, Real
from genlm.grammar.util import display_table


def test_nullary_fast():
    from examples import palindrome_ab

    batch_grammar = palindrome_ab.batch_grammar
    nullary_standard = batch_grammar.nullaryremove()
    nullary_fast = batch_grammar.nullaryremove_special()

    assert not nullary_fast.has_nullary()

    for string in ["","ab","aabb","aaaa"]:
        assert abs( nullary_standard(string) -nullary_fast(string)) < 1e-10
