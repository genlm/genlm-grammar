import warnings
from genlm.grammar import *  # pylint: disable=unused-import

warnings.warn(
    "Importing from `genlm_grammar` is deprecated. Please use `genlm.grammar` instead.",
    DeprecationWarning,
    stacklevel=2
)