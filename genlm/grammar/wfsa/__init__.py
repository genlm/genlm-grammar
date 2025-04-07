# import wfsa.base
from genlm.grammar.wfsa.field_wfsa import EPSILON, WFSA

one = WFSA.one
zero = WFSA.zero

__all__ = ["EPSILON", "WFSA", "one", "zero"]
