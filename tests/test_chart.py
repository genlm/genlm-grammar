from genlm.grammar.cfg import CFG
from genlm.grammar.semiring import Float, GradReal, Real
from genlm.grammar.chart import Chart, GradChart
import random

toll = 1e-12

def gr():
    return GradReal(random.uniform(-10,10))

def test_GradChart():
    a = gr()
    b = gr()
    c = gr()
    g_chart = GradChart(GradReal, {"a": a, "b": b, "c": c})
    r_chart = Chart(Real, {"a": a.to_real(), "b": b.to_real(), "c": c.to_real()})
    assert g_chart.product(["a", "b", "c"]).metric(g_chart.product(["a", "b", "c"])) < toll
    assert (g_chart + g_chart).metric(r_chart + r_chart) < toll
    assert (g_chart * g_chart).metric(r_chart * r_chart) < toll