from arsenal.maths import compare

from genlm.grammar.linear import WeightedGraph
from genlm.grammar.semiring import Float


def test_closure():
    G = WeightedGraph(Float)
    G["a", "a"] += 0.1
    G["a", "b"] += 0.5
    G["b", "a"] += 1
    G["b", "c"] += 1
    G["c", "d"] += 0.25
    G["d", "e"] += 1
    G["e", "c"] += 1
    G["e", "f"] += 1
    G["f", "f"] += 0.1
    G["f", "g"] += 1
    G["g", "g"] += 0.1

    want = G.closure_reference()
    have = G.closure_scc_based()

    c = compare(have, want)
    # c.show()
    assert c.max_err <= 1e-8

    # dry run of visualization methods
    import shutil

    graphviz_exists = shutil.which("dot") is not None
    if graphviz_exists:
        G._repr_svg_()


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
