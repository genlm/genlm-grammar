import re
import numpy as np
from genlm.grammar.chart import Chart


class Semiring:
    __slots__ = ("score",)

    def __init__(self, score):
        self.score = score

    @classmethod
    def chart(cls, *args, **kwargs):
        return Chart(cls, *args, **kwargs)

    #    @classmethod
    #    def zeros(cls, *shape):
    #        return np.full(shape, cls.zero)

    @classmethod
    def from_string(cls, x):
        return cls(float(x))

    def __add__(self, other):
        raise NotImplementedError()

    #    def __bool__(self):
    #        return self != self.zero

    def __mul__(self, other):
        raise NotImplementedError()

    #    def __float__(self):
    #        return float(self.score)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.score})"

    def __eq__(self, other):
        return isinstance(other, Semiring) and self.score == other.score

    #    def __lt__(self, other):
    #        raise NotImplementedError()
    #        return self.score < other.score

    #    def __hash__(self):
    #        raise NotImplementedError()
    #        return hash(self.score)

    def metric(self, other):
        return self != other


class Entropy(Semiring):
    def __init__(self, p, r):
        super().__init__((p, r))

    def star(self):
        tmp = 1 / (1 - self.score[0])
        return Entropy(tmp, tmp * tmp * self.score[1])

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return Entropy(self.score[0] + other.score[0], self.score[1] + other.score[1])

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Entropy(
            self.score[0] * other.score[0],
            self.score[0] * other.score[1] + self.score[1] * other.score[0],
        )

    @classmethod
    def from_string(cls, x):
        x = float(x)
        xlogx = x * np.log2(x) if x != 0 else 0
        return cls(x, xlogx)

    @property
    def H(self):
        p, r = self.score
        return np.log2(p) - r / p

    def metric(self, other):
        p1, r1 = self.score
        p2, r2 = other.score
        return max(abs(p1 - p2), abs(r1 - r2))


Entropy.zero = Entropy(0.0, 0.0)
Entropy.one = Entropy(1.0, 0.0)


class Boolean(Semiring):
    def __init__(self, x):
        super().__init__(bool(x))

    def star(self):
        return Boolean.one

    def __add__(self, other):
        if self.score or other.score:
            return Boolean.one
        else:
            return Boolean.zero

    def __mul__(self, other):
        if self.score and other.score:
            return Boolean.one
        else:
            return Boolean.zero

    def __repr__(self):
        return f"{self.score}"

    @classmethod
    def from_string(cls, x):
        x = x.strip()
        if x in {"True", "true", "1", "1.0"}:
            return Boolean.one
        else:
            assert x in {"False", "false", "0", "0.0"}, x
            return Boolean.zero


Boolean.zero = Boolean(False)
Boolean.one = Boolean(True)


class MaxPlus(Semiring):
    def star(self):
        return self.one

    def __add__(self, other):
        return MaxPlus(max(self.score, other.score))

    def __mul__(self, other):
        return MaxPlus(self.score + other.score)

    def metric(self, other):
        return abs(self.score - other.score)


MaxPlus.zero = MaxPlus(-np.inf)
MaxPlus.one = MaxPlus(0.0)


class MaxTimes(Semiring):
    def star(self):
        return self.one

    def __add__(self, other):
        return MaxTimes(max(self.score, other.score))

    def __mul__(self, other):
        return MaxTimes(self.score * other.score)

    def metric(self, other):
        return abs(self.score - other.score)


MaxTimes.zero = MaxTimes(0)
MaxTimes.one = MaxTimes(1)


class Float:
    def star(self):
        return 1 / (1 - self)

    @classmethod
    def from_string(cls, x):
        return float(x)

    def metric(x, y):  # pylint: disable=no-self-argument
        # if x == np.inf == y: return 0
        return abs(x - y)

    @classmethod
    def chart(cls, *args, **kwargs):
        return Chart(cls, *args, **kwargs)

    zero = 0
    one = 1


class Expectation(Semiring):
    def __init__(self, p1, p2):
        super().__init__((p1, p2))

    @classmethod
    def from_string(cls, x):
        try:
            y = re.findall(r"<(.*),\s*(.*)>", x)[0]
            return cls(float(y[0]), float(y[1]))
        except ValueError:
            raise AssertionError("Invalid input format. Expected '<float,float>'")  # pylint: disable=raise-missing-from

    def __add__(self, other):
        return Expectation(
            self.score[0] + other.score[0], self.score[1] + other.score[1]
        )

    def __mul__(self, other):
        return Expectation(
            self.score[0] * other.score[0],
            self.score[0] * other.score[1] + other.score[0] * self.score[1],
        )

    def star(self):
        p, r = self.score
        ps = 1 / (1 - p)
        return Expectation(ps, ps * r * ps)

    def __repr__(self):
        return f"<{self.score[0]},{self.score[1]}>"


Expectation.zero = Expectation.from_string("<0,0>")
Expectation.one = Expectation.from_string("<1,0>")


# class F128:
#    def star(self):
#        return 1 / (1 - self)
#
#    @classmethod
#    def from_string(cls, x):
#        return np.float128(x)
#
#    def metric(x, y):  # pylint: disable=no-self-argument
#        if x == np.inf == y:
#            return 0
#        return abs(x - y)
#
#    @classmethod
#    def chart(cls, *args, **kwargs):
#        return Chart(cls, *args, **kwargs)
#
#    zero = 0
#    one = 1
#
#
# getcontext().prec = 77
#
#
# class D256:
#    def star(self):
#        return 1 / (1 - self)
#
#    @classmethod
#    def from_string(cls, x):
#        return Decimal(x)
#
#    def metric(x, y):  # pylint: disable=no-self-argument
#        if x == np.inf == y:
#            return 0
#        return abs(x - y)
#
#    @classmethod
#    def chart(cls, *args, **kwargs):
#        return Chart(cls, *args, **kwargs)
#
#    zero = 0
#    one = 1


class Real(Semiring):
    def star(self):
        return Real(1 / (1 - self.score))

    def metric(self, other):
        return abs(self.score - other.score)

    def __add__(self, other):
        return Real(self.score + other.score)

    def __mul__(self, other):
        return Real(self.score * other.score)

    def __repr__(self):
        return f"{self.score}"


Real.zero = Real(0)
Real.one = Real(1)


class Log(Semiring):
    def metric(self, other):
        return abs(self.score - other.score)

    def star(self):
        return Log(-np.log1p(-np.exp(self.score)))

    def __add__(self, other):
        if self == Log.zero:
            return other
        if other == Log.zero:
            return self
        if self.score > other.score:
            return Log(self.score + np.log(1 + np.exp(other.score - self.score)))
        else:
            return Log(other.score + np.log(1 + np.exp(self.score - other.score)))

    def __mul__(self, other):
        if self == Log.zero:
            return Log.zero
        if other == Log.zero:
            return Log.zero
        return Log(self.score + other.score)


Log.zero = Log(-np.inf)
Log.one = Log(0.0)
