import numpy as np
from arsenal.maths import sample_dict


class LM:
    """Language model base class that defines a probability distribution over strings.

    A language model p: V* -> [0,1] defines a probability distribution over strings from
    a vocabulary V of tokens. Every language model admits a left-to-right factorization:

    p(x_1 x_2 ... x_T) = p(x_1|ε) p(x_2|x_1) ... p(x_T|x_1...x_{T-1}) p(EOS|x_1...x_T)

    Args:
        V: Vocabulary of symbols
        eos: Distinguished end-of-sequence symbol

    Attributes:
        V: Vocabulary set
        eos: End-of-sequence symbol

    Notes:
        Subclasses must implement p_next(xs) which returns p(·|x_1...x_T).
    """

    def __init__(self, V, eos):
        """Initialize language model with vocabulary and end-of-sequence token.

        Args:
            V: Vocabulary set of tokens
            eos: End-of-sequence token
        """
        self.eos = eos
        self.V = V

    def __call__(self, context):
        """Compute the probability of a complete string.

        Args:
            context: Sequence of tokens ending with eos token

        Returns:
            float: Probability of the sequence

        Raises:
            AssertionError: If context doesn't end with eos or contains invalid tokens
        """
        assert context[-1] == self.eos
        P = 1
        for i, y in enumerate(context):
            assert y in self.V, y
            p = self.p_next(context[:i])
            P *= p[y]
            if P == 0:
                break
        return P

    def logp(self, context):
        """Compute the log probability of a complete string.

        Args:
            context: Sequence of tokens ending with eos token

        Returns:
            (float): Log probability of the sequence

        Raises:
            AssertionError: If context doesn't end with eos
        """
        assert context[-1] == self.eos
        return sum(self.logp_next(context[:i])[y] for i, y in enumerate(context))

    def logp_next(self, context):
        """Compute the log conditional distribution over the next token given the prefix.

        Args:
            context: Sequence of tokens representing the prefix

        Returns:
            Log probabilities for each possible next token

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    def p_next(self, context):
        """Compute the conditional distribution over the next token given the prefix.

        Args:
            context: Sequence of tokens representing the prefix

        Returns:
            Probabilities for each possible next token

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    async def p_next_async(self, context):
        """Asynchronously compute the conditional distribution over the next token.

        Args:
            context: Sequence of tokens representing the prefix

        Returns:
            Probabilities for each possible next token
        """
        return self.p_next(context)

    def p_next_seq(self, context, extension):
        """Compute probability of an extension sequence given a context.

        Args:
            context: Sequence of tokens representing the prefix
            extension: Sequence of tokens to compute probability for

        Returns:
            (float): Probability of the extension sequence given the context

        Raises:
            AssertionError: If extension is empty
        """
        assert len(extension) >= 1
        P = 1
        for i in range(len(extension)):
            p = self.p_next(context + extension[:i])
            P *= p[extension[i]]
        return P

    def clear_cache(self):  # pragma: no cover
        """Clear any cached computations."""
        pass

    def sample(
        self,
        ys=(),
        draw=sample_dict,
        prob=True,
        verbose=0,
        max_tokens=np.inf,
        join=lambda ys, y: ys + (y,),
    ):
        """Sample a sequence from the language model.

        Args:
            ys: Initial sequence of tokens (default: empty tuple)
            draw: Function to sample from probability distribution (default: sample_dict)
            prob: Whether to return probability along with sequence (default: True)
            verbose: Verbosity level for printing tokens (default: 0)
            max_tokens: Maximum number of tokens to generate (default: infinity)
            join: Function to join new token with existing sequence (default: tuple concatenation)

        Returns:
            If prob=True: Tuple of (generated sequence, probability)
            If prob=False: Generated sequence
        """
        assert isinstance(ys, tuple), ys
        P = 1.0
        t = 0
        while True:
            p = self.p_next(ys).normalize()
            y = draw(p) if t <= max_tokens else self.eos
            P *= p[y]
            t += 1
            if verbose:
                if y == self.eos:
                    print()
                else:
                    print(y, end="")
            if y == self.eos:
                return (ys, P) if prob else ys
            ys = join(ys, y)
