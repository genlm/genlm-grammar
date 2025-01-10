import gc
import numpy as np
from arsenal.maths import sample_dict

from genlm_cfg.semiring import Float

class LM:
    r"""We say that $p\colon V^* \to [0,1]$ is a language model if $p$ is a probability
    distribution over strings from some alphabet $V$ of tokens.

    Every language model admits a left-to-right factorization:

    $$
    p(x_1 x_2 \cdots x_T) = p(x_1 \mid \varepsilon) p(x_2 \mid x_1) \cdots p(x_T \mid x_1 \cdots x_{T-1}) p(\mathrm{EOS} \mid x_1 \cdots x_T)
    $$

    Arguments:

      - `V`: a vocabulary of symbols

      - `eos`: a distinguished end of sequence symbol

      - `p_next(xs)`: $p(\cdot \mid x_1 \cdots x_T)$ is provided by subclasses.

    """

    def __init__(self, V, eos):
        self.eos = eos
        self.V = V

    def __call__(self, context):
        "Compute the probability of a complete string."
        # return np.exp(self.logp(ys))
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
        "Compute the probability of a complete string."
        assert context[-1] == self.eos
        return sum(self.logp_next(context[:i])[y] for i, y in enumerate(context))

    def logp_next(self, context):
        "Compute the log conditional distribution over the next token given the `prefix`."
        raise NotImplementedError()

    def p_next(self, context):
        "Compute the conditional distribution over the next token given the `prefix`."
        raise NotImplementedError()

    async def p_next_async(self, context):
        "Compute the (conditional) distribution over the next token given the `prefix`."
        return self.p_next(context)

    def p_next_seq(self, context, extension):
        """
        Compute `p(extension | context)` where `extension` is a sequence with |extension| > 1.
        """
        assert len(extension) >= 1
        P = 1
        for i in range(len(extension)):
            p = self.p_next(context + extension[:i])
            P *= p[extension[i]]
        return P

    def clear_cache(self):  # pragma: no cover
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
                    print(y, end='')
            if y == self.eos:
                return (ys, P) if prob else ys
            ys = join(ys, y)