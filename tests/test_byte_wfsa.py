import pytest
import string
import numpy as np
from genlm_cfg import Float, WFSA
from genlm_cfg.lark_interface import interegular_to_wfsa

def test_negated_pattern():
    wfsa = interegular_to_wfsa(r"[^abc]").to_bytes()
    
    assert wfsa(b"x") > 0
    assert wfsa(b"d") > 0
    assert wfsa(b"a") == 0
    assert wfsa(b"b") == 0
    assert wfsa(b"c") == 0

def test_emoji():
    wfsa = WFSA(Float)
    wfsa.add_I('S', 1)
    wfsa.add_F('F', 1)
    wfsa.add_arc('S', '👋', 'F', 1)

    wfsa = wfsa.to_bytes()
    
    wave_emoji = "👋".encode('utf-8')
    assert wfsa(wave_emoji) == 1
    assert wfsa(wave_emoji[:2]) == 0

def test_negated_pattern():
    pattern = r"[^👋a]"  # Matches anything except wave emoji or a
    wfsa = interegular_to_wfsa(
        pattern, charset=set(string.printable).union({"👋", "😊"})
    ).to_bytes()
    
    wave_emoji = "👋".encode('utf-8')
    
    assert wfsa(wave_emoji) == 0
    assert wfsa(b"a") == 0
    assert wfsa(b"x") > 0
    assert wfsa("😊".encode('utf-8')) > 0

def test_empty_pattern():
    wfsa = interegular_to_wfsa("").to_bytes()
    assert wfsa([]) > 0
    assert wfsa(b"a") == 0

def test_alternation_weights():
    # Test even weight distribution
    wfsa = interegular_to_wfsa("a|b|👋").to_bytes()
    weights = [wfsa(c) for c in [b'a', b'b', '👋'.encode()]]
    assert all(w == 1/3 for w in weights), weights

def test_byte_conversion_weights():
    # Test that converting to bytes preserves weights for multi-byte characters
    pattern = "👋|😊|x"

    wfsa = WFSA(Float)
    wfsa.add_I('S', 1)
    wfsa.add_F('F', 1)
    wfsa.add_arc('S', 'a', 'F', 1/3)
    wfsa.add_arc('S', "😊", 'F', 1/3)
    wfsa.add_arc('S', '👋', 'F', 1/3)

    byte_wfsa = wfsa.to_bytes()
    
    inputs = ['👋', '😊', 'x']
    for input_str in inputs:
        have = wfsa([input_str])
        want = byte_wfsa(input_str.encode('utf-8'))
        assert abs(have - want) < 1e-10, [have, want]

def test_cfg_conversion():
    wfsa = interegular_to_wfsa(r"a|b|👋|😊")
    bcfg = wfsa.to_bytes().to_cfg()
    assert bcfg(b"a") == wfsa(["b"])
    assert bcfg(b"b") == wfsa(["b"])
    assert bcfg("👋".encode('utf-8')) == wfsa(["👋"])
    assert bcfg("😊".encode('utf-8')) == wfsa(["😊"])