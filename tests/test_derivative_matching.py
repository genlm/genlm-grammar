from genlm_grammar.derivatives import (
    null,
    lazy,
    union,
    cat,
    chars,
    epsilon,
    matches,
    null,
    epsilon,
    dot,
    any,
)
from hypothesis import strategies as st, given, assume, settings, Phase
from hypothesis.stateful import rule, RuleBasedStateMachine


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
