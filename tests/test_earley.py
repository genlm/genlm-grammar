import pytest
from arsenal import colors

import examples
from genlm_cfg import add_EOS, EOS, CFG
from genlm_cfg.parse.earley import Earley
from genlm_cfg.parse.earley import Earley, EarleyLM
from genlm_cfg.parse.cky import CKYLM, IncrementalCKY
from genlm_cfg.semiring import Float, MaxTimes


def test_cycles():
    cfg = CFG.from_string(
        """
        0.5: S → A1
        0.5: S → A2

        0.5: A1 → B1
        0.5: B1 → C1
        0.5: C1 → A1

        0.5: A2 → B2
        0.5: B2 → C2
        0.5: C2 → A2

        1.0: C1 → C
        1.0: C2 → C

        0.5: C → c

        """,
        Float,
    )
    earley = Earley(cfg)
    assert_equal(earley('c'), cfg('c'))


def test_papa():
    cfg = examples.papa

    earley = Earley(cfg)

    x = 'papa ate the caviar'.split()
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10

    x = 'papa ate the caviar with the spoon'.split()
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10

    x = 'papa ate'.split()
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10


def test_papa_lm():
    cfg = examples.papa

    earley = EarleyLM(cfg)

    x = 'papa ate the caviar'.split()
    want = cfg(x)
    have = earley(x + [EOS])
    assert cfg.R.metric(have, want) <= 1e-10

    x = 'papa ate the caviar with the spoon'.split()
    want = cfg(x)
    have = earley(x + [EOS])
    assert cfg.R.metric(have, want) <= 1e-10

    x = 'papa ate'.split()
    want = cfg(x)
    have = earley(x + [EOS])
    assert cfg.R.metric(have, want) <= 1e-10


def test_palindrome():
    cfg = examples.palindrome_ab

    earley = Earley(cfg)

    x = ''
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10

    x = 'aabbaa'
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10

    x = 'aabba'
    want = cfg(x)
    have = earley(x)
    assert have == want == 0


def test_catalan():
    cfg = examples.catalan

    earley = Earley(cfg)

    x = ''
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10

    x = 'a'
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10

    x = 'aa'
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10

    x = 'aaaaa'
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10


def test_common_rhs():
    cfg = CFG.from_string(
        """
        0.1: S -> S S
        0.1: S -> S S
        0.8: S -> a
        """,
        Float,
    )

    cfg.agenda().assert_equal({x: 1 for x in cfg.N | cfg.V})

    earley = Earley(cfg)

    x = ''
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10, [x, have, want]

    x = 'a'
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10, [x, have, want]

    x = 'aa'
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10, [x, have, want]

    x = 'aaaaa'
    want = cfg(x)
    have = earley(x)
    assert cfg.R.metric(have, want) <= 1e-10


def test_has_unary_cycle():
    cfg = CFG.from_string(
        """
        0.5: S → A
        0.5: A → B
        0.5: B → C
        0.5: C → A
        0.5: C → c
        """,
        Float,
    )

    assert cfg.has_unary_cycle()

    cfg = CFG.from_string(
        """
        0.5: S → A
        0.5: A → B
        0.5: B → C
        0.5: C → c
        """,
        Float,
    )

    assert not cfg.has_unary_cycle()


def test_parse_unambiguous():
    cfg = CFG.from_string(
        """
        1.0: S → A B
        0.3: B → A B
        0.5: A → a
        0.4: B → b
        """,
        Float,
    )

    earley = Earley(cfg)
    assert_equal(earley('ab'), 0.2)
    assert_equal(earley('aab'), 0.03)
    assert_equal(earley('aaab'), 0.0045)


def test_parse_left_recursive():
    cfg = CFG.from_string(
        """
        1.0: S → A B
        0.3: A → A B
        0.5: A → a
        0.4: B → b
        """,
        Float,
    )

    earley = Earley(cfg)
    assert_equal(earley('ab'), 0.2)
    assert_equal(earley('abb'), 0.024)
    assert_equal(earley('abbb'), 0.00288)


def assert_equal(have, want, tol=1e-10):
    if isinstance(have, (float, int)):
        error = Float.metric(have, want)
    else:
        error = have.metric(want)
    assert error <= tol, f'have = {have}, want = {want}, error = {error}'


def test_parse_unary():
    # grammar contains non-cyclic unary rules
    cfg = CFG.from_string(
        """
        1.0: S → B
        0.3: B → A B
        0.2: B → A
        0.5: A → a
        """,
        Float,
    )

    earley = Earley(cfg)
    assert_equal(earley('a'), 0.1)
    assert_equal(earley('aa'), 0.015)
    assert_equal(earley('aaa'), 0.00225)

    cfg = CFG.from_string(
        """
        1.0: S → A
        0.5: S → c A
        0.3: A → B
        0.2: B → C
        0.5: C → c
        """,
        Float,
    )

    earley = Earley(cfg)
    assert_equal(earley('c'), 0.03)
    assert_equal(earley('cc'), 0.015)


def test_parse_mixed():
    cfg = CFG.from_string(
        """
        1.0: S → a B c D
        0.4: S → A b
        0.1: B → b b
        0.5: A → a
        0.3: D → d
        """,
        Float,
    )

    earley = Earley(cfg)
    assert_equal(earley('ab'), 0.2)
    assert_equal(earley('abbcd'), 0.03)


def test_parse_ambiguous_real():
    cfg = CFG.from_string(
        """
        1.0: S → A
        0.4: A → A + A
        0.1: A → A - A
        0.5: A → a
        """,
        Float,
    )

    earley = Earley(cfg)
    assert_equal(earley('a'), 0.5)
    assert_equal(earley('a+a'), 0.1)
    assert_equal(earley('a+a+a'), 0.04)

    cfg = CFG.from_string(
        """
        0.4: A → A + A
        0.1: A → A - A
        0.5: A → a
        """,
        Float,
        start='A',
    )

    earley = Earley(cfg)
    assert_equal(earley('a'), 0.5)
    assert_equal(earley('a+a'), 0.1)
    assert_equal(earley('a+a+a'), 0.04)


def test_parse_ambiguous_maxtimes():
    cfg = CFG.from_string(
        """
        1.0: S → A
        0.4: A → A + A
        0.1: A → A - A
        0.5: A → a
        """,
        MaxTimes,
    )

    earley = Earley(cfg)
    assert_equal(earley('a'), MaxTimes(0.5))
    assert_equal(earley('a+a'), MaxTimes(0.1))
    assert_equal(earley('a+a+a'), MaxTimes(0.02))


def test_p_next_new_abcdx():
    cfg = CFG.from_string(
        """
        1: S -> a b c d
        1: S -> a b c x
        1: S -> a b x x
        1: S -> a x x x
        1: S -> x x x x
        """,
        Float,
    )

    # Note: add_EOS used here for code coverage
    ckylm = CKYLM(add_EOS(cfg))
    earley = EarleyLM(add_EOS(cfg))

    for prefix in ['', 'a', 'ab', 'abc', 'abcd', 'acbd']:
        print()
        print(colors.light.blue % prefix)
        want = ckylm.p_next(prefix)
        print(want)
        have = earley.p_next(prefix)
        print(have)
        err = have.metric(want)
        print(colors.mark(err <= 1e-5))
        assert err <= 1e-5, err

    prefix = 'acbde'
    print()
    print(colors.light.blue % prefix)
    with pytest.raises(AssertionError):
        ckylm.p_next(prefix)
    with pytest.raises(AssertionError):
        earley.p_next(prefix)
    err = have.metric(want)
    print(colors.mark(err <= 1e-5))
    assert err <= 1e-5, err


def test_p_next_palindrome():
    cfg = examples.palindrome_ab

    ckylm = CKYLM(cfg)
    earley = EarleyLM(cfg)

    for prefix in ['', 'a', 'ab']:
        print()
        print(colors.light.blue % prefix)
        want = ckylm.p_next(prefix)
        print(want)
        have = earley.p_next(prefix)
        print(have)
        err = have.metric(want)
        print(colors.mark(err <= 1e-5))
        assert err <= 1e-5


def test_p_next_papa():
    cfg = examples.papa

    ckylm = CKYLM(cfg)
    earley = EarleyLM(cfg)

    for prefix in [
        [],
        ['papa'],
        ['papa', 'ate'],
        ['papa', 'ate', 'the'],
        ['papa', 'ate', 'the', 'caviar'],
    ]:
        prefix = tuple(prefix)
        print()
        print(colors.light.blue % (prefix,))
        want = ckylm.p_next(prefix)
        print(want)
        have = earley.p_next(prefix)
        print(have)
        print(colors.mark(have.metric(want) <= 1e-5))
        assert have.metric(want) <= 1e-5


def test_clear_cache():
    cfg = EarleyLM(examples.papa)
    assert len(cfg.model._chart) == 0
    sample = cfg.sample(prob=False) + (cfg.eos,)
    p = cfg(sample)
    assert len(cfg.model._chart) > 0
    print(p, sample)
    cfg.clear_cache()
    assert len(cfg.model._chart) == 0


# [2024-06-29 Sat] This test was added to improve code coverage. It does some
# out of the ordinary stuff (compute p_next without the prefix transformation).
def test_mystery():
    cfg = CFG.from_string(
        """
        1: S -> a b c d
        1: S -> a b c x
        1: S -> a b x x
        1: S -> a x x x
        1: S -> x x x x
        """,
        Float,
    )

    cky = IncrementalCKY(cfg.cnf)
    earley = Earley(cfg)

    for prefix in ['abc', 'abcd', '']:
        print()
        print(colors.light.blue % prefix)
        want = cky.p_next(prefix)
        print(want)
        have = earley.next_token_weights(earley.chart(prefix))
        print(have)
        err = have.metric(want)
        print(colors.mark(err <= 1e-5))
        assert err <= 1e-5, err


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
