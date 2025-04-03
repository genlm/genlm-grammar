from genlm.grammar.cfg import CFG
from genlm.grammar.semiring import Float

# simple CNF grammar, generates one string
abcd = CFG.from_string(
    """
1.0: S -> A BC
1.0: S -> A D
1.0: BC -> B C
1.0: A -> a
1.0: B -> b
1.0: C -> c
1.0: D -> d
""",
    Float,
)

# simple finite language where each prefix kills off a finite number of continuations
abcde_prefixes = CFG.from_string(
    """

1: S -> a b c d e
1: S -> a b c d x
1: S -> a b x x x
1: S -> a x x x x
1: S -> x x x x x

""",
    Float,
)

catalan = CFG.from_string(
    """
    0.2: S -> S S
    0.8: S -> a
    """,
    Float,
)

catalan_ab = CFG.from_string(
    """
    1.0: S -> A
    0.2: A -> A A
    0.4: A -> a
    0.4: A -> b
    """,
    Float,
)

palindrome_ab = CFG.from_string(
    """
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """,
    Float,
)

papa = CFG.from_string(
    """

    1: S    -> NP  VP

    0.2: NP   -> NP  PP
    0.7: NP   -> Det N
    0.1: NP   -> papa

    0.1: VP   -> V   NP
    0.1: VP   -> VP  PP
    0.8: VP   -> V

    1.0: PP   -> P   NP

    1: V -> ate
    1: Det -> the
    1: P -> with

    0.5: N -> caviar
    0.5: N -> spoon

    """,
    Float,
)
