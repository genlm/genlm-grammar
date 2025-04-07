from arsenal import colors
from genlm.grammar.util import format_table


class Chart(dict):
    """A weighted chart data structure that extends dict with semiring operations.

    The Chart class provides methods for semiring operations like addition and multiplication,
    as well as utilities for filtering, comparing, and manipulating weighted values.

    Attributes:
        semiring: The semiring that defines the weight operations
    """

    def __init__(self, semiring, vals=()):
        """Initialize a Chart.

        Args:
            semiring: The semiring for weight operations
            vals: Optional initial values for the chart
        """
        self.semiring = semiring
        super().__init__(vals)

    def __missing__(self, k):
        """Return zero weight for missing keys."""
        return self.semiring.zero

    def spawn(self):
        """Create a new empty Chart with the same semiring."""
        return Chart(self.semiring)

    def __add__(self, other):
        """Add two charts element-wise.

        Args:
            other: Another Chart to add to this one

        Returns:
            A new Chart containing the element-wise sum
        """
        new = self.spawn()
        for k, v in self.items():
            new[k] += v
        for k, v in other.items():
            new[k] += v
        return new

    def __mul__(self, other):
        """Multiply two charts element-wise.

        Args:
            other: Another Chart to multiply with this one

        Returns:
            A new Chart containing the element-wise product
        """
        new = self.spawn()
        for k in self:
            v = self[k] * other[k]
            if v == self.semiring.zero:
                continue
            new[k] += v
        return new

    def product(self, ks):
        """Compute the product of values for the given keys.

        Args:
            ks: Sequence of keys to multiply values for

        Returns:
            The product of values for the given keys
        """
        v = self.semiring.one
        for k in ks:
            v *= self[k]
        return v

    def copy(self):
        """Create a shallow copy of this Chart."""
        return Chart(self.semiring, self)

    def trim(self):
        """Return a new Chart with zero-weight entries removed."""
        return Chart(
            self.semiring, {k: v for k, v in self.items() if v != self.semiring.zero}
        )

    def metric(self, other):
        """Compute the maximum distance between this Chart and another.

        Args:
            other: Another Chart to compare against

        Returns:
            The maximum semiring metric between corresponding values
        """
        assert isinstance(other, Chart)
        err = 0
        for x in self.keys() | other.keys():
            err = max(err, self.semiring.metric(self[x], other[x]))
        return err

    def _repr_html_(self):
        """Return HTML representation for Jupyter notebooks."""
        return (
            '<div style="font-family: Monospace;">'
            + format_table(self.trim().items(), headings=["key", "value"])
            + "</div>"
        )

    def __repr__(self):
        """Return string representation, excluding zero weights."""
        return repr({k: v for k, v in self.items() if v != self.semiring.zero})

    def __str__(self, style_value=lambda k, v: str(v)):
        """Return formatted string representation.

        Args:
            style_value: Optional function to format values

        Returns:
            Formatted string showing non-zero entries
        """

        def key(k):
            return -self.semiring.metric(self[k], self.semiring.zero)

        return (
            "Chart {\n"
            + "\n".join(
                f"  {k!r}: {style_value(k, self[k])},"
                for k in sorted(self, key=key)
                if self[k] != self.semiring.zero
            )
            + "\n}"
        )

    def assert_equal(self, want, *, domain=None, tol=1e-5, verbose=False, throw=True):
        """Assert that this Chart equals another within tolerance.

        Args:
            want: The expected Chart or dict of values
            domain: Optional set of keys to check
            tol: Tolerance for floating point comparisons
            verbose: Whether to print detailed comparison
            throw: Whether to raise AssertionError on mismatch
        """
        if not isinstance(want, Chart):
            want = self.semiring.chart(want)
        if domain is None:
            domain = self.keys() | want.keys()
        assert verbose or throw
        errors = []
        for x in domain:
            if self.semiring.metric(self[x], want[x]) <= tol:
                if verbose:
                    print(colors.mark(True), x, self[x])
            else:
                if verbose:
                    print(colors.mark(False), x, self[x], want[x])
                errors.append(x)
        if throw:
            for x in errors:
                raise AssertionError(f"{x}: {self[x]} {want[x]}")

    def argmax(self):
        """Return the key with maximum value."""
        return max(self, key=self.__getitem__)

    def argmin(self):
        """Return the key with minimum value."""
        return min(self, key=self.__getitem__)

    def top(self, k):
        """Return a new Chart with the k largest values.

        Args:
            k: Number of top values to keep

        Returns:
            A new Chart containing only the k largest values
        """
        return Chart(
            self.semiring,
            {k: self[k] for k in sorted(self, key=self.__getitem__, reverse=True)[:k]},
        )

    def max(self):
        """Return the maximum value in the Chart."""
        return max(self.values())

    def min(self):
        """Return the minimum value in the Chart."""
        return min(self.values())

    def sum(self):
        """Return the sum of all values in the Chart."""
        return sum(self.values())

    def sort(self, **kwargs):
        """Return a new Chart with entries sorted by key.

        Args:
            **kwargs: Arguments passed to sorted()

        Returns:
            A new Chart with sorted entries
        """
        return self.semiring.chart((k, self[k]) for k in sorted(self, **kwargs))

    def sort_descending(self):
        """Return a new Chart with entries sorted by decreasing value."""
        return self.semiring.chart(
            (k, self[k]) for k in sorted(self, key=lambda k: -self[k])
        )

    def normalize(self):
        """Return a new Chart with values normalized to sum to 1."""
        Z = self.sum()
        if Z == 0:
            return self
        return self.semiring.chart((k, v / Z) for k, v in self.items())

    def filter(self, f):
        """Return a new Chart keeping only entries where f(key) is True.

        Args:
            f: Predicate function that takes a key and returns bool

        Returns:
            A new Chart containing only entries where f(key) is True
        """
        return self.semiring.chart((k, v) for k, v in self.items() if f(k))

    def project(self, f):
        """Apply a function to keys, summing weights when transformed keys overlap.

        Args:
            f: Function to transform keys

        Returns:
            A new Chart with transformed keys and summed weights
        """
        out = self.semiring.chart()
        for k, v in self.items():
            out[f(k)] += v
        return out

    # TODO: the more general version of this method is join
    def compare(self, other, *, domain=None):
        """Compare this Chart to another using pandas DataFrame.

        Args:
            other: Another Chart or dict to compare against
            domain: Optional set of keys to compare

        Returns:
            pandas DataFrame showing key-by-key comparison
        """
        import pandas as pd

        if not isinstance(other, Chart):
            other = self.semiring.chart(other)
        if domain is None:
            domain = self.keys() | other.keys()
        rows = []
        for x in domain:
            m = self.semiring.metric(self[x], other[x])
            rows.append(dict(key=x, self=self[x], other=other[x], metric=m))
        return pd.DataFrame(rows)
