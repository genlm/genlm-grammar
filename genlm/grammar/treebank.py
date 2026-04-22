from genlm.grammar import CFG 
from genlm.grammar.semiring import Float
import re

class TreebankCFG(CFG):
    """
    Weighted Context-free Grammar

    A weighted context-free grammar consists of:\n
    - `R`: A semiring that defines the weights\n
    - `S`: A start symbol (nonterminal)\n
    - `V`: A set of terminal symbols (vocabulary)\n
    - `N`: A set of nonterminal symbols\n
    - `rules`: A list of weighted production rules\n

    Each rule has the form: w: X -> Y1 Y2 ... Yn where:\n
    - w is a weight from the semiring R\n
    - X is a nonterminal symbol\n
    - Y1...Yn are terminal or nonterminal symbols\n
    """


    @classmethod
    def from_string(
        cls,
        string,
        comment="#",
        start="ROOT",
        is_terminal=lambda x: x[0] == '_',
    ):
        """
        Create a CFG from a string representation.

        Args:
            string: The grammar rules as a string
            semiring: The semiring for rule weights
            comment: Comment character to ignore lines (default: '#')
            start: Start symbol (default: 'S')
            is_terminal: Function to identify terminal symbols (default: lowercase first letter)

        Returns:
            A new CFG instance
        """
        V = set()
        V.add('_UNK')
        cfg = cls(Float, S=start, V=V)
        for line in string.split("\n"):
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            try:
                # Support optional leading number (e.g. "1 Discourse->[...] : 0.03")
                [(lhs, rhs, w)] = re.findall(r"(?:[\d.eE+\-]+\s+)?(\S+)->\[\s*(.*)\]\s*:\s*(.*)$", line)
                lhs = lhs.strip()
                rhs = rhs.strip().split()
                for x in rhs:
                    if is_terminal(x):
                        V.add(x)
                cfg.add(Float.from_string(w), lhs, *rhs)
            except ValueError:
                raise ValueError(f"bad input line:\n{line}")  # pylint: disable=W0707
        return cfg



    def replace_unknown(self, sentence, count=True):
        if isinstance(sentence, str):
            tokens = sentence.strip().split()
        else:
            tokens = sentence

        replaced = []
        assert '_UNK' in self.V, "'_UNK' not in vocabulary"
        count = 0

        for token in tokens:
            if token in self.V:
                replaced.append(token)
            elif '_UNK' in self.V:
                replaced.append('_UNK')
                count += 1

        return replaced, count