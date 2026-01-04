from genlm.grammar import CFG, Real, Boolean, Float
from genlm.grammar.cfg import Rule
from genlm.grammar.semiring import GradReal
from genlm.grammar.chart import Chart, GradChart

from genlm.grammar.wfsa import EPSILON
from itertools import product 


def test_gradient():
    grammar_string = """
    0.9: S → A B C
    1.3: A → a
    1.1: B → b
    0.7: C → C
    0.3: C →
    """
    
    eps = 1e-7
    
    # Gradient with Pytorch.
    cfg = CFG.from_string(grammar_string, GradReal)
    cfg.enable_grad()
    result = cfg.nullaryremove()
    result.treesum().backward()
    analytical = {i: r.w.grad_item() for i, r in enumerate(cfg.rules)}
    
    # Gradient approximation with finite differences  
    numerical = {}
    for i in range(len(cfg.rules)):
        cfg_plus = CFG.from_string(grammar_string, GradReal)
        cfg_plus.rules[i].w.score += eps # Add +eps to all the weight rules.
        f_plus = cfg_plus.nullaryremove().treesum().score.item()
        
        cfg_minus = CFG.from_string(grammar_string, GradReal)
        cfg_minus.rules[i].w.score -= eps # Add -eps to all the weight rules.
        f_minus = cfg_minus.nullaryremove().treesum().score.item()
        
        numerical[i] = (f_plus - f_minus) / (2 * eps) # Compuet the finite difference approximation of the gradient.
    
    # Compare
    for i, r in enumerate(cfg.rules):
        ana = analytical[i] if analytical[i] is not None else 0.0
        num = numerical[i]
        assert abs(ana - num) < 1e-5, "{r}: ana={ana:.4f}, num={num:.4f}"
        print(f"{r}: ana={ana:.4f}, num={num:.4f}")

