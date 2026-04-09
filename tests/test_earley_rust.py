"""
Tests for the Rust Earley implementation (EarleyRust / EarleyRustLM).

Mirrors the Python Earley tests in test_earley.py, ensuring the Rust backend
produces identical results for parsing, next-token weights, and the LM interface.
"""

import pytest
from arsenal import colors

import examples
from genlm.grammar import add_EOS, EOS, CFG
from genlm.grammar.semiring import Float
from genlm.grammar.parse.earley import Earley, EarleyLM
from genlm.grammar.parse.cky import CKYLM, IncrementalCKY

try:
    from genlm.grammar.parse.earley_rust import EarleyRust, EarleyRustLM

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
# Parsing tests: EarleyRust.__call__ vs Python Earley and brute-force
# ────────────────────────────────────────────────────────────────────


class TestRustParsing:
    """EarleyRust.__call__ should match Python Earley on various grammars."""

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
        rust = EarleyRust(cfg)
        python = Earley(cfg)
        assert_equal(rust("c"), python("c"))

    def test_papa(self):
        cfg = examples.papa
        rust = EarleyRust(cfg)

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
        rust = EarleyRust(cfg)

        for x in ["", "aabbaa", "aabba"]:
            want = cfg(x)
            have = rust(x)
            assert_equal(have, want)

    def test_catalan(self):
        cfg = examples.catalan
        rust = EarleyRust(cfg)

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
        rust = EarleyRust(cfg)

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
        rust = EarleyRust(cfg)
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
        rust = EarleyRust(cfg)
        assert_equal(rust("ab"), 0.2)
        assert_equal(rust("abb"), 0.024)
        assert_equal(rust("abbb"), 0.00288)

    def test_parse_unary(self):
        cfg = CFG.from_string(
            """
            1.0: S → A
            0.5: A → B
            0.3: A → a
            0.5: B → b
            """,
            Float,
        )
        rust = EarleyRust(cfg)
        assert_equal(rust("a"), 0.3)
        assert_equal(rust("b"), 0.25)

    def test_parse_mixed(self):
        cfg = CFG.from_string(
            """
            0.5: S → A B
            0.5: S → a
            0.5: A → a
            0.5: B → b
            """,
            Float,
        )
        rust = EarleyRust(cfg)
        assert_equal(rust("a"), 0.5)
        assert_equal(rust("ab"), 0.125)


# ────────────────────────────────────────────────────────────────────
# Rust vs Python agreement on parsing
# ────────────────────────────────────────────────────────────────────


class TestRustVsPythonParsing:
    """EarleyRust and Python Earley should produce identical parse weights."""

    @pytest.mark.parametrize(
        "cfg_name",
        ["papa", "catalan", "catalan_ab", "palindrome_ab", "abcd", "abcde_prefixes"],
    )
    def test_agreement(self, cfg_name):
        cfg = getattr(examples, cfg_name)
        rust = EarleyRust(cfg)
        python = Earley(cfg)

        # Generate test strings from the grammar
        strings = _generate_strings(cfg, max_len=6)
        assert len(strings) > 0, f"No test strings generated for {cfg_name}"

        for x in strings:
            have = rust(x)
            want = python(x)
            assert_equal(have, want), f"Disagreement on {cfg_name}, input={x}"


# ────────────────────────────────────────────────────────────────────
# Next-token weights: Rust vs Python Earley
# ────────────────────────────────────────────────────────────────────


class TestRustNextTokenWeights:
    """EarleyRust.next_token_weights should match Python Earley."""

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
        rust = EarleyRust(prefix_cfg)
        python = Earley(prefix_cfg)

        for prefix in ["", "a", "ab", "abc", "abcd"]:
            rust_ntw = rust.next_token_weights(rust.chart(prefix))
            python_ntw = python.next_token_weights(python.chart(prefix))
            err = rust_ntw.metric(python_ntw)
            assert err <= 1e-5, f"prefix={prefix!r}, err={err}"

    def test_next_token_palindrome(self):
        cfg = examples.palindrome_ab
        prefix_cfg = cfg.prefix_grammar
        rust = EarleyRust(prefix_cfg)
        python = Earley(prefix_cfg)

        for prefix in ["", "a", "ab", "aba", "abab"]:
            rust_ntw = rust.next_token_weights(rust.chart(prefix))
            python_ntw = python.next_token_weights(python.chart(prefix))
            err = rust_ntw.metric(python_ntw)
            assert err <= 1e-5, f"prefix={prefix!r}, err={err}"

    def test_next_token_catalan(self):
        cfg = examples.catalan
        prefix_cfg = cfg.prefix_grammar
        rust = EarleyRust(prefix_cfg)
        python = Earley(prefix_cfg)

        for prefix in ["", "a", "aa", "aaa"]:
            rust_ntw = rust.next_token_weights(rust.chart(prefix))
            python_ntw = python.next_token_weights(python.chart(prefix))
            err = rust_ntw.metric(python_ntw)
            assert err <= 1e-5, f"prefix={prefix!r}, err={err}"

    def test_next_token_papa(self):
        cfg = examples.papa
        prefix_cfg = cfg.prefix_grammar
        rust = EarleyRust(prefix_cfg)
        python = Earley(prefix_cfg)

        for prefix in [
            [],
            ["papa"],
            ["papa", "ate"],
            ["papa", "ate", "the"],
            ["papa", "ate", "the", "caviar"],
            ["papa", "ate", "the", "caviar", "with"],
        ]:
            rust_ntw = rust.next_token_weights(rust.chart(prefix))
            python_ntw = python.next_token_weights(python.chart(prefix))
            err = rust_ntw.metric(python_ntw)
            assert err <= 1e-5, f"prefix={prefix}, err={err}"

    @pytest.mark.parametrize(
        "cfg_name",
        ["papa", "catalan", "catalan_ab", "palindrome_ab"],
    )
    def test_next_token_vs_cky(self, cfg_name):
        """Rust next-token weights should also match CKY (independent reference)."""
        cfg = getattr(examples, cfg_name)
        prefix_cfg = cfg.prefix_grammar
        rust = EarleyRust(prefix_cfg)
        cky = IncrementalCKY(prefix_cfg.cnf)

        strings = _generate_strings(cfg, max_len=5)
        # Use prefixes of generated strings
        prefixes = set()
        for s in strings:
            for i in range(len(s) + 1):
                prefixes.add(s[:i])

        for prefix in prefixes:
            rust_ntw = rust.next_token_weights(rust.chart(prefix))
            cky_ntw = cky.p_next(prefix)
            err = rust_ntw.metric(cky_ntw)
            assert err <= 1e-5, f"{cfg_name}, prefix={prefix}, err={err}"


# ────────────────────────────────────────────────────────────────────
# LM interface: EarleyRustLM vs EarleyLM and CKYLM
# ────────────────────────────────────────────────────────────────────


class TestRustLM:
    """EarleyRustLM.p_next should match EarleyLM and CKYLM."""

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
        rust_lm = EarleyRustLM(cfg_eos)
        python_lm = EarleyLM(cfg_eos)
        cky_lm = CKYLM(cfg_eos)

        for prefix in ["", "a", "ab", "abc", "abcd"]:
            rust_p = rust_lm.p_next(prefix)
            python_p = python_lm.p_next(prefix)
            cky_p = cky_lm.p_next(prefix)
            assert rust_p.metric(python_p) <= 1e-5, f"vs Python, prefix={prefix!r}"
            assert rust_p.metric(cky_p) <= 1e-5, f"vs CKY, prefix={prefix!r}"

    def test_p_next_palindrome(self):
        cfg = examples.palindrome_ab
        rust_lm = EarleyRustLM(cfg)
        python_lm = EarleyLM(cfg)

        for prefix in ["", "a", "ab"]:
            rust_p = rust_lm.p_next(prefix)
            python_p = python_lm.p_next(prefix)
            err = rust_p.metric(python_p)
            assert err <= 1e-5, f"prefix={prefix!r}, err={err}"

    def test_p_next_papa(self):
        cfg = examples.papa
        rust_lm = EarleyRustLM(cfg)
        python_lm = EarleyLM(cfg)

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

    def test_p_next_catalan(self):
        cfg = examples.catalan
        rust_lm = EarleyRustLM(cfg)
        python_lm = EarleyLM(cfg)

        for prefix in ["", "a", "aa", "aaa"]:
            rust_p = rust_lm.p_next(prefix)
            python_p = python_lm.p_next(prefix)
            err = rust_p.metric(python_p)
            assert err <= 1e-5, f"prefix={prefix!r}, err={err}"

    def test_full_string_weight(self):
        """LM should recover the full string weight via the chain rule."""
        cfg = examples.papa
        rust_lm = EarleyRustLM(cfg)

        x = "papa ate the caviar".split()
        want = cfg(x)
        have = rust_lm(x + [EOS])
        assert cfg.R.metric(have, want) <= 1e-10

    def test_sample(self):
        """Sampling should produce valid strings."""
        cfg = examples.catalan
        rust_lm = EarleyRustLM(cfg)

        for _ in range(10):
            sample = rust_lm.sample(prob=False) + (rust_lm.eos,)
            p = rust_lm(sample)
            assert p > 0, f"sampled string {sample} has zero weight"


# ────────────────────────────────────────────────────────────────────
# Cache behavior
# ────────────────────────────────────────────────────────────────────


class TestRustCache:

    def test_clear_cache(self):
        cfg = examples.papa
        rust_lm = EarleyRustLM(cfg)

        # Parse something to populate the cache
        sample = rust_lm.sample(prob=False) + (rust_lm.eos,)
        p = rust_lm(sample)
        assert p > 0

        # Clear and re-parse: should get same result
        rust_lm.clear_cache()
        p2 = rust_lm(sample)
        assert_equal(p, p2)

    def test_cache_consistency(self):
        """Results should be the same whether chart is cached or freshly computed."""
        cfg = examples.papa
        prefix_cfg = cfg.prefix_grammar
        rust = EarleyRust(prefix_cfg)

        prefix = ["papa", "ate"]

        # First call: populates cache
        ntw1 = rust.next_token_weights(rust.chart(prefix))

        # Second call: uses cache
        ntw2 = rust.next_token_weights(rust.chart(prefix))

        assert ntw1.metric(ntw2) <= 1e-10

        # After clearing: should still agree
        rust.clear_cache()
        ntw3 = rust.next_token_weights(rust.chart(prefix))
        assert ntw1.metric(ntw3) <= 1e-10


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _generate_strings(cfg, max_len=6):
    """Generate strings from the grammar up to a given length using brute force."""
    from itertools import product

    terminals = sorted(cfg.V)
    strings = []
    for length in range(max_len + 1):
        for combo in product(terminals, repeat=length):
            x = combo if any(isinstance(t, str) and " " in t for t in terminals) else combo
            w = cfg(x)
            if w > 0:
                strings.append(x)
    return strings
