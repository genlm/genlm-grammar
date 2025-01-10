"""
Algorithms for solving left-linear or right-linear systems of equations over closed semirings.
"""

import html
from collections import defaultdict
from functools import cached_property

from arsenal import Integerizer
from graphviz import Digraph


class WeightedGraph:
    """
    Weight graph equipped with efficient methods for solving algebraic path
    problems.
    """

    def __init__(self, WeightType):
        self.WeightType = WeightType
        self.N = set()
        self.incoming = defaultdict(set)
        self.outgoing = defaultdict(set)
        self.E = WeightType.chart()

    def __iter__(self):
        return iter(self.E)

    def __getitem__(self, item):
        i, j = item
        return self.E[i, j]

    def __setitem__(self, item, value):
        i, j = item
        self.N.add(i)
        self.N.add(j)
        if value != self.WeightType.zero:
            self.E[i, j] = value
            self.incoming[j].add(i)
            self.outgoing[i].add(j)
        return self

    def closure(self):
        C = WeightedGraph(self.WeightType)
        K = self.closure_scc_based()
        for i, j in K:
            C[i, j] += K[i, j]
        return C

    def closure_reference(self):
        return self._closure(self.E, self.N)

    def closure_scc_based(self):
        K = self.WeightType.chart()
        for i in self.N:
            b = self.WeightType.chart()
            b[i] = self.WeightType.one
            sol = self.solve_left(b)
            for j in sol:
                K[i, j] = sol[j]
        return K

    def solve_left(self, b):
        """
        Solve `x = x A + b` using block, upper-triangular decomposition.
        """
        sol = self.WeightType.chart()
        for block, B in self.Blocks:
            # Compute the total weight of entering the block from the left at
            # each entry j in the block
            enter = self.WeightType.chart()
            for j in block:
                enter[j] += b[j]
                for i in self.incoming[j]:
                    enter[j] += sol[i] * self.E[i, j]

            # Now, compute the total weight of completing the block
            for j, k in B:
                sol[k] += enter[j] * B[j, k]

        return sol

    def solve_right(self, b):
        """
        Solve `x = A x + b` using block, upper-triangular decomposition.
        """
        sol = self.WeightType.chart()
        for block, B in reversed(self.Blocks):
            # Compute the total weight of entering the block from the right at
            # each entry point j in the block
            enter = self.WeightType.chart()
            for j in block:
                enter[j] += b[j]
                for k in self.outgoing[j]:
                    enter[j] += self.E[j, k] * sol[k]

            # Now, compute the total weight of completing the block
            for i, j in B:
                sol[i] += B[i, j] * enter[j]

        return sol

    def _closure(self, A, N):
        """
        Compute the reflexive, transitive closure of `A` for the block of nodes `N`.
        """

        # Special handling for the common case of |N| = 1; XXX: I'm surprised
        # how much faster this version is compared to the loops below it.
        if len(N) == 1:
            [i] = N
            return {(i, i): self.WeightType.star(self.E[i, i])}

        A = self.E
        old = A.copy()
        new = self.WeightType.chart()
        # transitive closure
        for j in N:
            new.clear()
            sjj = self.WeightType.star(old[j, j])
            for i in N:
                for k in N:
                    new[i, k] = old[i, k] + old[i, j] * sjj * old[j, k]
            old, new = new, old
        # reflexive closure
        for i in N:
            old[i, i] += self.WeightType.one
        return old

    @cached_property
    def blocks(self):
        "List of blocks."
        return list(self._blocks(self.N))

    def _blocks(self, roots=None):
        return scc_decomposition(self.incoming.__getitem__, roots)

    @cached_property
    def buckets(self):
        "Map from node to block id."
        return {x: i for i, block in enumerate(self.blocks) for x in block}

    @cached_property
    def Blocks(self):
        return [(block, self._closure(self.E, block)) for block in self.blocks]

    def _repr_svg_(self):
        return self.graphviz()._repr_image_svg_xml()

    def graphviz(self, label_format=str, escape=lambda x: html.escape(str(x))):
        "Convert to `graphviz.Digraph` instance for visualization."
        name = Integerizer()

        g = Digraph(
            node_attr=dict(
                fontname='Monospace',
                fontsize='9',
                height='0',
                width='0',
                margin='0.055,0.042',
                penwidth='0.15',
                shape='box',
                style='rounded',
            ),
            edge_attr=dict(
                penwidth='0.5',
                arrowhead='vee',
                arrowsize='0.5',
                fontname='Monospace',
                fontsize='8',
            ),
        )

        for i, j in self:
            # if self.E[i, j] == self.WeightType.zero:
            #    continue
            g.edge(str(name(i)), str(name(j)), label=label_format(self.E[i, j]))

        for i in self.N:
            g.node(str(name(i)), label=escape(i))

        return g


def scc_decomposition(successors, roots):
    r"""
    Find the strongly connected components of a graph.
    Implemention is based on Tarjan's (1972) algorithm; runs in $\mathcal{O}(E + V)$ time, uses $\mathcal{O}(V)$ space.

    Tarjan, R. E. (1972), [Depth-first search and linear graph algorithms](https://epubs.siam.org/doi/10.1137/0201010)
    SIAM Journal on Computing, 1 (2): 146–160, doi:10.1137/0201010

    """

    # 'Low Link Value' of a node is the smallest id reachable by DFS, including itself.
    # low link values are initialized to each node's id.
    lowest = {}  # node -> position of the root of the SCC

    stack = []  # stack
    trail = set()  # set of nodes on the stack
    t = 0

    def dfs(v):
        # DFS pushes nodes onto the stack
        nonlocal t
        t += 1
        num = t
        lowest[v] = t
        trail.add(v)
        stack.append(v)

        for w in successors(v):
            if lowest.get(w) is None:
                # As usual, only recurse when we haven't already visited this node
                yield from dfs(w)
                # The recursive call will find a cycle if there is one.
                # `lowest` is used to propagate the position of the earliest
                # node on the cycle in the DFS.
                lowest[v] = min(lowest[v], lowest[w])
            elif w in trail:
                # Collapsing cycles.  If `w` comes before `v` in dfs and `w` is
                # on the stack, then we've detected a cycle and we can start
                # collapsing values in the SCC.  It might not be the maximal
                # SCC. The min and stack will take care of that.
                lowest[v] = min(lowest[v], lowest[w])

        if lowest[v] == num:
            # `v` is the root of an SCC; We're totally done with that subgraph.
            # nodes above `v` on the stack are an SCC.
            C = []
            while True:  # pop until we reach v
                w = stack.pop()
                trail.remove(w)
                C.append(w)
                if w == v:
                    break
            yield frozenset(C)

    for v in roots:
        if lowest.get(v) is None:
            yield from dfs(v)
