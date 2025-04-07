import warnings
from genlm.grammar import *  # noqa: F403

warnings.warn(
    "Importing from `genlm_grammar` is deprecated. Please use `genlm.grammar` instead.",
    DeprecationWarning,
    stacklevel=2,
)
