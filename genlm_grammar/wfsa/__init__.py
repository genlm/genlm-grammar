# import wfsa.base
from genlm_grammar.wfsa.field_wfsa import EPSILON, WFSA

one = WFSA.one
zero = WFSA.zero

__all__ = ["EPSILON", "WFSA", "one", "zero"]
