"""
Tests for the Rust rescaled Earley implementation.

Mirrors the Python rescaled Earley tests in test_earley_rescaled.py,
ensuring the Rust backend produces identical results for parsing,
next-token weights, logp, and the LM interface.
"""

import pytest
import numpy as np
from arsenal import colors

import examples
from genlm.grammar import add_EOS, EOS, CFG
from genlm.grammar.semiring import Float
from genlm.grammar.parse.earley_rescaled import Earley as EarleyRescaledPython
from genlm.grammar.parse.earley_rescaled import EarleyLM as EarleyLMRescaledPython
from genlm.grammar.parse.cky import CKYLM, IncrementalCKY

try:
    from genlm.grammar.parse.earley_rescaled_rust import (
        EarleyRescaledRust,
        EarleyRescaledRustLM,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension (genlm_earley) not compiled"
)


def assert_equal(have, want, tol=1e-10):
    if isinstance(have, (float, int)):
        error = Float.metric(have, want)
    else:
        error = have.metric(want)
    assert error <= tol, f"have = {have}, want = {want}, error = {error}"


# ────────────────────────────────────────────────────────────────────
# Rescaling-specific: logp on long strings that would underflow
# ────────────────────────────────────────────────────────────────────


class TestRescaling:

    def test_basics_long_string(self):
        """The key rescaling test: a very long string that would underflow without rescaling."""
        cfg = CFG.from_string(
            """
            0.001: S → a S
            0.999: S → a
            """,
            Float,
        )
        rust_lm = EarleyRescaledRustLM(cfg)

        x = "a" * 110
        want = (len(x) - 1) * np.log(0.001) + 1 * np.log(0.999)
        assert Float.metric(rust_lm.model.logp(x + EOS), want) <= 1e-5

        p = rust_lm.p_next(x)
        p.assert_equal({"a": 0.001, EOS: 0.999}, tol=1e-10)

    def test_logp_vs_python(self):
        """Rust logp should match Python rescaled logp."""
        cfg = CFG.from_string(
            """
            0.001: S → a S
            0.999: S → a
            """,
            Float,
        )
        rust = EarleyRescaledRust(add_EOS(cfg).prefix_grammar)
        python = EarleyRescaledPython(add_EOS(cfg).prefix_grammar)

        for length in [10, 50, 110]:
            x = "a" * length + EOS
            rust_logp = rust.logp(x)
            python_logp = python.logp(x)
            assert abs(rust_logp - python_logp) <= 1e-5, (
                f"length={length}, rust={rust_logp}, python={python_logp}"
            )


# ────────────────────────────────────────────────────────────────────
# Parsing tests: EarleyRescaledRust.__call__ vs brute-force
# ────────────────────────────────────────────────────────────────────


class TestRustRescaledParsing:

    def test_cycles(self):
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
        rust = EarleyRescaledRust(cfg)
        assert_equal(rust("c"), cfg("c"))

    def test_papa(self):
        cfg = examples.papa
        rust = EarleyRescaledRust(cfg)

        for x in [
            "papa ate the caviar".split(),
            "papa ate the caviar with the spoon".split(),
            "papa ate".split(),
        ]:
            want = cfg(x)
            have = rust(x)
            assert cfg.R.metric(have, want) <= 1e-10, f"failed on {x}"

    def test_palindrome(self):
        cfg = examples.palindrome_ab
        rust = EarleyRescaledRust(cfg)

        for x in ["", "aabbaa", "aabba"]:
            want = cfg(x)
            have = rust(x)
            assert_equal(have, want)

    def test_catalan(self):
        cfg = examples.catalan
        rust = EarleyRescaledRust(cfg)

        for x in ["", "a", "aa", "aaaaa"]:
            want = cfg(x)
            have = rust(x)
            assert_equal(have, want)

    def test_common_rhs(self):
        cfg = CFG.from_string(
            """
            0.1: S -> S S
            0.1: S -> S S
            0.8: S -> a
            """,
            Float,
        )
        rust = EarleyRescaledRust(cfg)

        for x in ["", "a", "aa", "aaaaa"]:
            want = cfg(x)
            have = rust(x)
            assert_equal(have, want)

    def test_parse_unambiguous(self):
        cfg = CFG.from_string(
            """
            1.0: S → A B
            0.3: B → A B
            0.5: A → a
            0.4: B → b
            """,
            Float,
        )
        rust = EarleyRescaledRust(cfg)
        assert_equal(rust("ab"), 0.2)
        assert_equal(rust("aab"), 0.03)
        assert_equal(rust("aaab"), 0.0045)

    def test_parse_left_recursive(self):
        cfg = CFG.from_string(
            """
            1.0: S → A B
            0.3: A → A B
            0.5: A → a
            0.4: B → b
            """,
            Float,
        )
        rust = EarleyRescaledRust(cfg)
        assert_equal(rust("ab"), 0.2)
        assert_equal(rust("abb"), 0.024)
        assert_equal(rust("abbb"), 0.00288)

    def test_parse_unary(self):
        cfg = CFG.from_string(
            """
            1.0: S → B
            0.3: B → A B
            0.2: B → A
            0.5: A → a
            """,
            Float,
        )
        rust = EarleyRescaledRust(cfg)
        assert_equal(rust("a"), 0.1)
        assert_equal(rust("aa"), 0.015)
        assert_equal(rust("aaa"), 0.00225)

    def test_parse_mixed(self):
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
        rust = EarleyRescaledRust(cfg)
        assert_equal(rust("ab"), 0.2)
        assert_equal(rust("abbcd"), 0.03)

    def test_parse_ambiguous_real(self):
        cfg = CFG.from_string(
            """
            1.0: S → A
            0.4: A → A + A
            0.1: A → A - A
            0.5: A → a
            """,
            Float,
        )
        rust = EarleyRescaledRust(cfg)
        assert_equal(rust("a"), 0.5)
        assert_equal(rust("a+a"), 0.1)
        assert_equal(rust("a+a+a"), 0.04)


# ────────────────────────────────────────────────────────────────────
# Rust rescaled vs Python rescaled agreement
# ────────────────────────────────────────────────────────────────────


class TestRustVsPythonRescaledParsing:

    @pytest.mark.parametrize(
        "cfg_name",
        ["papa", "catalan", "catalan_ab", "palindrome_ab", "abcd", "abcde_prefixes"],
    )
    def test_agreement(self, cfg_name):
        cfg = getattr(examples, cfg_name)
        rust = EarleyRescaledRust(cfg)
        python = EarleyRescaledPython(cfg)

        strings = _generate_strings(cfg, max_len=6)
        assert len(strings) > 0, f"No test strings generated for {cfg_name}"

        for x in strings:
            have = rust(x)
            want = python(x)
            assert_equal(have, want), f"Disagreement on {cfg_name}, input={x}"


# ────────────────────────────────────────────────────────────────────
# Next-token weights: Rust rescaled vs Python rescaled and CKY
# ────────────────────────────────────────────────────────────────────


class TestRustRescaledNextTokenWeights:

    def test_next_token_abcdx(self):
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
        prefix_cfg = cfg.prefix_grammar
        rust = EarleyRescaledRust(prefix_cfg)
        python = EarleyRescaledPython(prefix_cfg)

        for prefix in ["", "a", "ab", "abc", "abcd"]:
            rust_ntw = rust.next_token_weights(rust.chart(prefix))
            python_ntw = python.next_token_weights(python.chart(prefix))
            err = rust_ntw.metric(python_ntw)
            assert err <= 1e-5, f"prefix={prefix!r}, err={err}"

    def test_next_token_palindrome(self):
        cfg = examples.palindrome_ab
        prefix_cfg = cfg.prefix_grammar
        rust = EarleyRescaledRust(prefix_cfg)
        python = EarleyRescaledPython(prefix_cfg)

        for prefix in ["", "a", "ab", "aba", "abab"]:
            rust_ntw = rust.next_token_weights(rust.chart(prefix))
            python_ntw = python.next_token_weights(python.chart(prefix))
            err = rust_ntw.metric(python_ntw)
            assert err <= 1e-5, f"prefix={prefix!r}, err={err}"

    def test_next_token_papa(self):
        cfg = examples.papa
        prefix_cfg = cfg.prefix_grammar
        rust = EarleyRescaledRust(prefix_cfg)
        python = EarleyRescaledPython(prefix_cfg)

        for prefix in [
            [],
            ["papa"],
            ["papa", "ate"],
            ["papa", "ate", "the"],
            ["papa", "ate", "the", "caviar"],
        ]:
            rust_ntw = rust.next_token_weights(rust.chart(prefix))
            python_ntw = python.next_token_weights(python.chart(prefix))
            err = rust_ntw.metric(python_ntw)
            assert err <= 1e-5, f"prefix={prefix}, err={err}"


# ────────────────────────────────────────────────────────────────────
# LM interface: EarleyRescaledRustLM vs EarleyLMRescaledPython and CKYLM
# ────────────────────────────────────────────────────────────────────


class TestRustRescaledLM:

    def test_p_next_abcdx(self):
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
        cfg_eos = add_EOS(cfg)
        rust_lm = EarleyRescaledRustLM(cfg_eos)
        python_lm = EarleyLMRescaledPython(cfg_eos)
        cky_lm = CKYLM(cfg_eos)

        for prefix in ["", "a", "ab", "abc", "abcd"]:
            rust_p = rust_lm.p_next(prefix)
            python_p = python_lm.p_next(prefix)
            cky_p = cky_lm.p_next(prefix).normalize()
            assert rust_p.metric(python_p) <= 1e-5, f"vs Python, prefix={prefix!r}"
            assert rust_p.metric(cky_p) <= 1e-5, f"vs CKY, prefix={prefix!r}"

    def test_p_next_palindrome(self):
        cfg = examples.palindrome_ab
        rust_lm = EarleyRescaledRustLM(cfg)
        python_lm = EarleyLMRescaledPython(cfg)

        for prefix in ["", "a", "ab"]:
            rust_p = rust_lm.p_next(prefix)
            python_p = python_lm.p_next(prefix)
            err = rust_p.metric(python_p)
            assert err <= 1e-5, f"prefix={prefix!r}, err={err}"

    def test_p_next_papa(self):
        cfg = examples.papa
        rust_lm = EarleyRescaledRustLM(cfg)
        python_lm = EarleyLMRescaledPython(cfg)

        for prefix in [
            [],
            ["papa"],
            ["papa", "ate"],
            ["papa", "ate", "the"],
            ["papa", "ate", "the", "caviar"],
        ]:
            prefix = tuple(prefix)
            rust_p = rust_lm.p_next(prefix)
            python_p = python_lm.p_next(prefix)
            err = rust_p.metric(python_p)
            assert err <= 1e-5, f"prefix={prefix}, err={err}"

    def test_full_string_weight(self):
        cfg = examples.papa
        rust_lm = EarleyRescaledRustLM(cfg)

        x = "papa ate the caviar".split()
        want = cfg(x)
        have = rust_lm(x + [EOS])
        assert cfg.R.metric(have, want) <= 1e-10

    def test_sample(self):
        cfg = examples.catalan
        rust_lm = EarleyRescaledRustLM(cfg)

        for _ in range(10):
            sample = rust_lm.sample(prob=False) + (rust_lm.eos,)
            p = rust_lm(sample)
            assert p > 0, f"sampled string {sample} has zero weight"


# ────────────────────────────────────────────────────────────────────
# Cache behavior
# ────────────────────────────────────────────────────────────────────


class TestRustRescaledCache:

    def test_clear_cache(self):
        cfg = examples.papa
        rust_lm = EarleyRescaledRustLM(cfg)
        sample = rust_lm.sample(prob=False) + (rust_lm.eos,)
        p = rust_lm(sample)
        assert p > 0
        rust_lm.clear_cache()
        p2 = rust_lm(sample)
        assert_equal(p, p2)

    def test_cache_consistency(self):
        cfg = examples.papa
        prefix_cfg = cfg.prefix_grammar
        rust = EarleyRescaledRust(prefix_cfg)

        prefix = ["papa", "ate"]
        ntw1 = rust.next_token_weights(rust.chart(prefix))
        ntw2 = rust.next_token_weights(rust.chart(prefix))
        assert ntw1.metric(ntw2) <= 1e-10

        rust.clear_cache()
        ntw3 = rust.next_token_weights(rust.chart(prefix))
        assert ntw1.metric(ntw3) <= 1e-10


# ────────────────────────────────────────────────────────────────────
# Mystery test (next_token_weights without prefix grammar)
# ────────────────────────────────────────────────────────────────────


class TestRustRescaledMystery:

    def test_mystery(self):
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
        rust = EarleyRescaledRust(cfg)

        for prefix in ["abc", "abcd", ""]:
            want = cky.p_next(prefix)
            want = want.normalize() if want.trim() else want
            have = rust.next_token_weights(rust.chart(prefix))
            err = have.metric(want)
            assert err <= 1e-5, f"prefix={prefix!r}, err={err}"


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _generate_strings(cfg, max_len=6):
    from itertools import product

    terminals = sorted(cfg.V)
    strings = []
    for length in range(max_len + 1):
        for combo in product(terminals, repeat=length):
            w = cfg(combo)
            if w > 0:
                strings.append(combo)
    return strings
