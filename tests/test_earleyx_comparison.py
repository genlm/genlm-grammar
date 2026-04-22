"""
Tests verifying that our prefix parser agrees with EarleyX (Luong et al.,
TACL 2013) on normalized (proper PCFG) toy grammars.

EarleyX implement Stolcke's (1995) prefix probability algorithm.
Stolcke's formulation assumes the grammar is a proper PCFG (inside(X) = 1
for all nonterminals). 

Our algorithm, obtained via the prefix grammar and standard Earley parsing, handles also un-normalized grammars.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
from genlm.grammar.cfg import CFG
from genlm.grammar.semiring import Float
from genlm.grammar.parse.earley import Earley
from genlm.grammar.cfglm import locally_normalize

EARLEYX_DIR = Path(__file__).resolve().parent.parent / "earleyx"
EARLEYX_CLASSPATH = f"{EARLEYX_DIR / 'classes'}:{EARLEYX_DIR / 'lib'}/*"

# EarleyX is vendored in the repo — fail loudly if not compiled
assert (EARLEYX_DIR / "classes" / "parser" / "Main.class").exists(), (
    f"EarleyX not compiled. Run 'ant compile' in {EARLEYX_DIR}"
)


def make_parser(grammar_str, start='ROOT'):
    """Build prefix Earley parser from grammar string."""
    cfg = CFG.from_string(grammar_str, Float, start=start)
    return Earley(cfg.prefix_grammar), cfg


def cfg_to_earleyx_grammar(grammar_str):
    """Convert our grammar string format to EarleyX's format.

    Ours:   0.6: NN -> dog
    Theirs: NN->[_dog] : 0.6

    EarleyX distinguishes terminals from nonterminals by the '_' prefix.
    We detect terminals as RHS symbols that never appear as a LHS.
    """
    rules = []
    nonterminals = set()
    for line in grammar_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'([\d.]+)\s*:\s*(\S+)\s*->\s*(.+)', line)
        if m:
            prob, lhs, rhs = m.group(1), m.group(2), m.group(3).strip()
            nonterminals.add(lhs)
            rules.append((prob, lhs, rhs.split()))

    lines = []
    for prob, lhs, rhs_tokens in rules:
        converted = [t if t in nonterminals else f'_{t}' for t in rhs_tokens]
        lines.append(f"{lhs}->[{' '.join(converted)}] : {prob}")
    return '\n'.join(lines)


def run_earleyx(grammar_str, sentences, start='ROOT'):
    """Run EarleyX prefix parser and return per-sentence prefix probabilities.

    Returns dict: sentence_index -> list of (word, prefix_prob) pairs,
    one per word position.
    """
    earleyx_grammar = cfg_to_earleyx_grammar(grammar_str)

    with tempfile.TemporaryDirectory() as tmpdir:
        grammar_file = os.path.join(tmpdir, "grammar.txt")
        input_file = os.path.join(tmpdir, "input.txt")
        out_prefix = os.path.join(tmpdir, "result")

        with open(grammar_file, 'w') as f:
            f.write(earleyx_grammar + '\n')

        with open(input_file, 'w') as f:
            for sent in sentences:
                f.write(' '.join(sent) + '\n')

        cmd = [
            "java", "-classpath", EARLEYX_CLASSPATH,
            "parser.Main",
            "-in", input_file,
            "-out", out_prefix,
            "-grammar", grammar_file,
            "-obj", "prefix",
            "-root", start,
            "-normalprob",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"EarleyX failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        output_file = f"{out_prefix}.prefix"
        results = {}
        current_sent = None
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#!') or line.startswith('# !'):
                    continue
                if line.startswith('#'):
                    try:
                        current_sent = int(line[1:].strip())
                        if current_sent not in results:
                            results[current_sent] = []
                    except ValueError:
                        continue
                elif current_sent is not None:
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            results[current_sent].append(
                                (parts[0], float(parts[1]))
                            )
                        except ValueError:
                            continue

        return results


def assert_prefixes_match(grammar_str, sentences, start='ROOT', tol=1e-6):
    """Assert that our parser and EarleyX agree on all prefix probabilities."""
    parser, _ = make_parser(grammar_str, start=start)
    earleyx = run_earleyx(grammar_str, sentences, start=start)

    mismatches = []
    for i, sent in enumerate(sentences):
        details = earleyx.get(i, [])
        assert len(details) == len(sent), (
            f"EarleyX returned {len(details)} values for {len(sent)}-word sentence {i}"
        )
        for j in range(1, len(sent) + 1):
            ours = float(parser(sent[:j]))
            theirs = details[j - 1][1]
            if abs(ours - theirs) > tol:
                mismatches.append((i, sent[:j], ours, theirs))

    if mismatches:
        msg = "Prefix probability mismatches:\n"
        for si, prefix, ours, theirs in mismatches:
            msg += (
                f"  sent {si}, prefix {prefix}: "
                f"ours={ours:.10e}, EarleyX={theirs:.10e}\n"
            )
        pytest.fail(msg)


class TestEarleyXAgreement:
    """On normalized (proper PCFG) toy grammars, our prefix parser and
    EarleyX must produce identical prefix probabilities at every position."""

    def test_simple_np_vp(self):
        """Simple NP-VP grammar with determiners, nouns, and verbs."""
        grammar = '''
            0.9: ROOT -> S
            0.1: ROOT -> NP
            0.8: S -> NP VP
            0.2: S -> VP
            0.6: NP -> DT NN
            0.4: NP -> NN
            0.7: VP -> VB
            0.3: VP -> VB NP
            1.0: DT -> the
            0.5: NN -> dog
            0.5: NN -> cat
            0.6: VB -> runs
            0.4: VB -> sees
        '''
        sentences = [
            ('the', 'dog', 'runs'),
            ('the', 'cat', 'sees', 'the', 'dog'),
            ('dog', 'runs'),
            ('cat',),
        ]
        assert_prefixes_match(grammar, sentences)

    def test_longer_rhs(self):
        """Grammar with longer RHS rules (ternary productions)."""
        grammar = '''
            1.0: ROOT -> S
            0.5: S -> NP VP PP
            0.5: S -> NP VP
            0.6: NP -> DT NN
            0.4: NP -> NN
            0.7: VP -> VB
            0.3: VP -> VB NP
            1.0: PP -> IN NP
            1.0: DT -> the
            0.5: NN -> dog
            0.5: NN -> cat
            0.6: VB -> runs
            0.4: VB -> sees
            1.0: IN -> with
        '''
        sentences = [
            ('the', 'dog', 'runs', 'with', 'the', 'cat'),
            ('dog', 'sees', 'the', 'cat'),
        ]
        assert_prefixes_match(grammar, sentences)

    def test_unary_chain(self):
        """Grammar with a unary chain: ROOT -> S -> A -> terminal."""
        grammar = '''
            1.0: ROOT -> S
            0.4: S -> A
            0.6: S -> a b
            1.0: A -> c
        '''
        sentences = [
            ('a', 'b'),
            ('c',),
        ]
        assert_prefixes_match(grammar, sentences)

    def test_deeply_ambiguous(self):
        """Highly ambiguous grammar: multiple derivations for the same string."""
        grammar = '''
            1.0: ROOT -> S
            0.4: S -> S S
            0.3: S -> NP VP
            0.3: S -> VP
            0.5: NP -> DT NN
            0.5: NP -> NN
            0.6: VP -> VB
            0.4: VP -> VB NP
            1.0: DT -> the
            0.5: NN -> a
            0.5: NN -> b
            0.6: VB -> a
            0.4: VB -> b
        '''
        sentences = [
            ('a',),
            ('a', 'b'),
            ('the', 'a', 'a'),
            ('a', 'b', 'a'),
        ]
        assert_prefixes_match(grammar, sentences)

    def test_multiple_preterminals(self):
        """Grammar where a terminal is generated by multiple preterminals."""
        grammar = '''
            1.0: ROOT -> S
            0.5: S -> NP VP
            0.5: S -> VP
            0.5: NP -> NN
            0.5: NP -> JJ NN
            0.6: VP -> VB
            0.4: VP -> VB NP
            0.5: NN -> fish
            0.5: NN -> dog
            1.0: JJ -> big
            0.6: VB -> fish
            0.4: VB -> run
        '''
        sentences = [
            ('fish',),
            ('fish', 'fish'),
            ('big', 'fish', 'run'),
            ('fish', 'big', 'dog'),
        ]
        assert_prefixes_match(grammar, sentences)


class TestLocallyNormalize:
    """Test that locally_normalize produces a proper PCFG and preserves
    string probabilities (up to a global constant)."""

    def _assert_is_proper_pcfg(self, cfg, tol=1e-10):
        """Check that inside(X) ≈ 1 for all nonterminals."""
        Z = cfg.agenda()
        for nt in cfg.N:
            assert abs(float(Z[nt]) - 1.0) < tol, (
                f"inside({nt}) = {float(Z[nt])}, expected 1.0"
            )

    def _assert_string_probs_proportional(self, orig, norm, sentences, start='S', tol=1e-6):
        """Check that string probabilities under orig and norm are proportional."""
        orig_parser = Earley(orig)
        norm_parser = Earley(norm)

        orig_probs = [float(orig_parser(s)) for s in sentences]
        norm_probs = [float(norm_parser(s)) for s in sentences]

        # Find the ratio from the first nonzero pair
        ratio = None
        for o, n in zip(orig_probs, norm_probs):
            if o > 1e-30 and n > 1e-30:
                ratio = o / n
                break

        assert ratio is not None, "No nonzero string probabilities found"

        for i, (o, n) in enumerate(zip(orig_probs, norm_probs)):
            if o < 1e-30 and n < 1e-30:
                continue
            assert abs(o - n * ratio) < tol * abs(o), (
                f"String {sentences[i]}: orig={o}, norm={n}, "
                f"expected ratio={ratio}, got {o/n if n > 0 else 'inf'}"
            )

    def test_already_normalized(self):
        """A proper PCFG should be unchanged by locally_normalize."""
        cfg = CFG.from_string('''
            0.6: S -> NP VP
            0.4: S -> VP
            0.5: NP -> DT NN
            0.5: NP -> NN
            0.7: VP -> VB
            0.3: VP -> VB NP
            1.0: DT -> the
            0.5: NN -> dog
            0.5: NN -> cat
            0.6: VB -> runs
            0.4: VB -> sees
        ''', Float, start='S')

        norm = locally_normalize(cfg)
        self._assert_is_proper_pcfg(norm)

        # Weights should be (nearly) unchanged
        orig_weights = {(r.head, tuple(r.body)): float(r.w) for r in cfg}
        for r in norm:
            key = (r.head, tuple(r.body))
            assert abs(float(r.w) - orig_weights[key]) < 1e-10, (
                f"Rule {key}: orig={orig_weights[key]}, norm={float(r.w)}"
            )

    def test_unnormalized_preterminals(self):
        """Preterminal weights sum > 1: NN -> dog (0.6) + NN -> cat (0.5) = 1.1"""
        cfg = CFG.from_string('''
            0.8: S -> NP VP
            0.2: S -> VP
            0.6: NP -> DT NN
            0.4: NP -> NN
            0.7: VP -> VB
            0.3: VP -> VB NP
            1.0: DT -> the
            0.6: NN -> dog
            0.5: NN -> cat
            0.6: VB -> runs
            0.4: VB -> sees
        ''', Float, start='S')

        norm = locally_normalize(cfg)
        self._assert_is_proper_pcfg(norm)

        sentences = [('the', 'dog', 'runs'), ('cat', 'sees', 'the', 'dog')]
        self._assert_string_probs_proportional(cfg, norm, sentences)

    def test_unary_cycle(self):
        """Unary cycle S -> A -> S with weights such that cycle weight < 1.

        locally_normalize should converge in one round.
        """
        cfg = CFG.from_string('''
            0.6: S -> A
            0.5: S -> a b
            0.3: A -> S
            0.8: A -> c
        ''', Float, start='S')

        norm = locally_normalize(cfg)
        self._assert_is_proper_pcfg(norm)

        sentences = [('a', 'b'), ('c',)]
        self._assert_string_probs_proportional(cfg, norm, sentences)

    def test_unary_cycle_high_weight(self):
        """Unary cycle S -> A -> S with cycle weight close to 1 (0.9 * 0.95 = 0.855).

        The geometric series 1/(1 - 0.855) ≈ 6.9, so inside weights are large
        but finite. locally_normalize should still produce a proper PCFG.
        """
        cfg = CFG.from_string('''
            0.9: S -> A
            0.5: S -> a b
            0.95: A -> S
            0.8: A -> c
        ''', Float, start='S')

        norm = locally_normalize(cfg)
        self._assert_is_proper_pcfg(norm)

        sentences = [('a', 'b'), ('c',)]
        self._assert_string_probs_proportional(cfg, norm, sentences)

    def test_normalized_unnormalized_grammar_matches_earleyx(self):
        """After locally_normalize, an unnormalized grammar should produce
        prefix probabilities that match EarleyX."""
        grammar_str = '''
            0.9: ROOT -> S
            0.1: ROOT -> NP
            0.8: S -> NP VP
            0.2: S -> VP
            0.6: NP -> DT NN
            0.4: NP -> NN
            0.7: VP -> VB
            0.3: VP -> VB NP
            1.0: DT -> the
            0.6: NN -> dog
            0.5: NN -> cat
            0.6: VB -> runs
            0.4: VB -> sees
        '''

        cfg = CFG.from_string(grammar_str, Float, start='ROOT')
        norm = locally_normalize(cfg)

        # Reconstruct grammar string from normalized CFG
        norm_lines = []
        for r in norm:
            norm_lines.append(f"{float(r.w)}: {r.head} -> {' '.join(r.body)}")
        norm_str = '\n'.join(norm_lines)

        sentences = [
            ('the', 'dog', 'runs'),
            ('the', 'cat', 'sees', 'the', 'dog'),
            ('dog', 'runs'),
        ]
        assert_prefixes_match(norm_str, sentences, start='ROOT')




class TestEarleyXPerWordTiming:
    """Verify that EarleyX emits per-word cumulative timing (## WordTime lines)."""

    GRAMMAR = '''
        0.9: ROOT -> S
        0.1: ROOT -> NP
        0.8: S -> NP VP
        0.2: S -> VP
        0.6: NP -> DT NN
        0.4: NP -> NN
        0.7: VP -> VB
        0.3: VP -> VB NP
        1.0: DT -> the
        0.5: NN -> dog
        0.5: NN -> cat
        0.6: VB -> runs
        0.4: VB -> sees
    '''

    def test_wordtime_lines_emitted(self):
        """EarleyX should emit one ## WordTime line per word position."""
        earleyx_grammar = cfg_to_earleyx_grammar(self.GRAMMAR)
        sentences = [('the', 'dog', 'runs'), ('cat', 'sees', 'the', 'dog')]

        with tempfile.TemporaryDirectory() as tmpdir:
            grammar_file = os.path.join(tmpdir, "grammar.txt")
            input_file = os.path.join(tmpdir, "input.txt")
            out_prefix = os.path.join(tmpdir, "result")

            with open(grammar_file, 'w') as f:
                f.write(earleyx_grammar + '\n')
            with open(input_file, 'w') as f:
                for sent in sentences:
                    f.write(' '.join(sent) + '\n')

            cmd = [
                "java", "-classpath", EARLEYX_CLASSPATH,
                "parser.Main",
                "-in", input_file,
                "-out", out_prefix,
                "-grammar", grammar_file,
                "-obj", "prefix",
                "-root", "ROOT",
                "-normalprob",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, f"EarleyX failed: {result.stderr[:500]}"

            # Parse ## WordTime lines from stderr
            pattern = re.compile(r"## WordTime (\S+) (\d+) ([\d.]+)")
            word_times = []
            for m in pattern.finditer(result.stderr):
                sent_id = m.group(1)
                word_pos = int(m.group(2))
                cumulative_ms = float(m.group(3))
                word_times.append((sent_id, word_pos, cumulative_ms))

            # Should have one line per word position per sentence
            assert len(word_times) == 3 + 4, (
                f"Expected 7 WordTime lines (3 + 4 words), got {len(word_times)}"
            )

            # Cumulative times should be monotonically increasing within each sentence
            by_sent = {}
            for sent_id, word_pos, ms in word_times:
                by_sent.setdefault(sent_id, []).append((word_pos, ms))

            for sent_id, times in by_sent.items():
                for i in range(1, len(times)):
                    assert times[i][1] >= times[i-1][1], (
                        f"Sentence {sent_id}: cumulative time not monotonic: "
                        f"{times[i-1]} -> {times[i]}"
                    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
