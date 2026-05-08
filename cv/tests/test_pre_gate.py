"""Tests for cv.pre_gate (spec §10.3 + DR v3.0.9 §7)."""
import math

import numpy as np

from cv.pre_gate import aggregate_pre_gate, class_prior_entropy, pre_gate_ratio


def test_entropy_uniform_3class():
    """H([0,1,2,0,1,2]) → log(3)."""
    H = class_prior_entropy(np.array([0, 1, 2, 0, 1, 2]), n_classes=3)
    assert abs(H - math.log(3)) < 1e-9


def test_entropy_skewed():
    """H([0,0,0,0,1,2]) hand-computed: p=[4/6, 1/6, 1/6]."""
    p = [4 / 6, 1 / 6, 1 / 6]
    expected = -sum(pi * math.log(pi) for pi in p)
    H = class_prior_entropy(np.array([0, 0, 0, 0, 1, 2]), n_classes=3)
    assert abs(H - expected) < 1e-9


def test_entropy_single_class_zero():
    """All same class → entropy 0 (prior is degenerate)."""
    H = class_prior_entropy(np.array([1, 1, 1, 1, 1]), n_classes=3)
    assert abs(H) < 1e-12


def test_pre_gate_ratio():
    train = np.array([0, 1, 2, 0, 1, 2])  # uniform → H = log(3) ≈ 1.0986
    val_logloss = 1.05
    expected = 1.05 / math.log(3)
    assert abs(pre_gate_ratio(val_logloss, train) - expected) < 1e-9


def test_aggregate_pass_4_of_6():
    ratios = [0.95, 0.97, 0.95, 0.99, 0.98, 1.01]  # first 4 < 0.99
    res = aggregate_pre_gate(ratios, threshold=0.99, required_pass=4)
    assert res["passed"] is True
    assert res["n_passed"] == 4


def test_aggregate_fail_3_of_6():
    ratios = [0.95, 0.97, 0.95, 1.00, 1.01, 1.02]  # only 3 < 0.99
    res = aggregate_pre_gate(ratios, threshold=0.99, required_pass=4)
    assert res["passed"] is False
    assert res["n_passed"] == 3


def test_aggregate_all_pass():
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    res = aggregate_pre_gate(ratios, threshold=0.99, required_pass=4)
    assert res["passed"] is True
    assert res["n_passed"] == 6
