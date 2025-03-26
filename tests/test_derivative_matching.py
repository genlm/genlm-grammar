from genlm_grammar.derivatives import (
    lazy,
    union,
    cat,
    char,
    chars,
    epsilon,
    matches,
    null,
    epsilon,
    dot,
    any,
    literal,
    seq,
    optional,
    derivative,
)
from hypothesis import strategies as st, given, assume, settings, Phase, example
from hypothesis.stateful import rule, RuleBasedStateMachine
from hypothesis.extra.lark import from_lark
from lark import Lark
import string
import pytest


settings.register_profile("default", phases=set(settings().phases) - {Phase.explain})


def test_resolves_lazy_self_reference():
    x = lazy(lambda: x)
    assert x.value is null


def test_resolves_lazy_cycles():
    x = lazy(lambda: y)
    y = lazy(lambda: x)
    assert x.value is null
    assert y.has_been_forced
    assert y.value is null


def test_union_with_null_is_ignored():
    assert union(epsilon, null) == epsilon


@st.composite
def base_grammar(draw):
    recur = base_grammar()
    return draw(
        st.one_of(
            st.sampled_from([null, epsilon, dot]),
            st.builds(chars, st.frozensets(st.integers(0, 255))),
            st.builds(any, st.integers(2, 10)),
            st.builds(
                cat,
                recur,
                recur,
            ),
            st.builds(union, recur, recur),
        )
    )


@st.composite
def base_grammar_matching_empty(draw):
    any_grammar = base_grammar()
    recur = base_grammar_matching_empty()
    return draw(
        st.one_of(
            st.just(epsilon),
            st.builds(
                cat,
                recur,
                recur,
            ),
            st.builds(union, recur, any_grammar),
        )
    )


@given(base_grammar_matching_empty(), base_grammar())
def test_union_match_empty(g1, g2):
    assume(g1.matches_empty)
    assert union(g1, g2).matches_empty


@given(base_grammar_matching_empty())
def test_should_match_empty(g):
    assert g.matches_empty
    assert g.could_have_matches


@given(st.from_regex(b"^[01]*$").map(lambda s: s.strip()))
def test_sequence_matching(s):
    x = union(epsilon, cat(chars(b"01"), lazy(lambda: x)))

    assert matches(x, s)


JSON_LARK_GRAMMAR = R"""
?value: dict
        | list
        | string
        | SIGNED_NUMBER      -> number
        | "true"             -> true
        | "false"            -> false
        | "null"             -> null

list : "[" [value ("," value)*] "]"

dict : "{" [pair ("," pair)*] "}"
pair : string ":" value

string : ESCAPED_STRING

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"""

JSON_STRATEGY = from_lark(Lark(JSON_LARK_GRAMMAR, start="value"))

WHITESPACE = seq(chars(frozenset(string.whitespace.encode("ascii"))))

JSON_VALUE = (
    WHITESPACE
    + lazy(
        lambda: union(
            JSON_DICT,
            JSON_LIST,
            JSON_STRING,
            SIGNED_NUMBER,
            literal(b"true"),
            literal(b"false"),
            literal(b"null"),
        )
    )
    + WHITESPACE
)


JSON_STRING = cat(
    literal(b'"'),
    seq(
        union(
            chars(set(range(256)) - set(b'"\\')),
            cat(literal(b"\\"), dot),
        )
    ),
    literal(b'"'),
)


JSON_LIST = cat(
    literal(b"["),
    WHITESPACE,
    optional(
        cat(
            JSON_VALUE,
            WHITESPACE,
            seq(
                cat(
                    literal(b","),
                    WHITESPACE,
                    JSON_VALUE,
                )
            ),
        )
    ),
    WHITESPACE,
    literal(b"]"),
)

KV_PAIR = cat(
    JSON_STRING,
    WHITESPACE,
    literal(b":"),
    WHITESPACE,
    JSON_VALUE,
)

JSON_DICT = cat(
    literal(b"{"),
    WHITESPACE,
    optional(
        cat(
            KV_PAIR,
            WHITESPACE,
            seq(
                cat(
                    literal(b","),
                    WHITESPACE,
                    KV_PAIR,
                )
            ),
        )
    ),
    WHITESPACE,
    literal(b"}"),
)

DIGIT = chars((b"0123456789"))
HEXDIGIT = chars((b"0123456789abcdefABCDEF"))

SIGN = optional(chars(b"+-"))

INT = DIGIT + seq(DIGIT)
SIGNED_INT = SIGN + INT

DECIMAL = union(cat(INT, literal(b"."), optional(INT)), cat(literal(b"."), INT))

_EXP = chars(b"eE") + SIGNED_INT
FLOAT = (INT + _EXP) | (DECIMAL + optional(_EXP))

SIGNED_FLOAT = SIGN + FLOAT

NUMBER = FLOAT | INT
SIGNED_NUMBER = SIGN + NUMBER


@pytest.mark.parametrize(
    "literal",
    [b'"\\\\"'],
)
def test_string_literals(literal):
    assert matches(JSON_STRING, literal)


@example('"\\\\"')
@example("{} ")
@settings(deadline=None)
@given(JSON_STRATEGY)
def test_any_json_from_lark_matches(json):
    assert matches(JSON_VALUE, json.encode("utf-8"))
    print("Done")
