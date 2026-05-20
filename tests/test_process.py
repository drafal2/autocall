"""Tests for the StochasticProcessArray builder."""

import numpy as np
import pytest

from autocall.process import build_process_array


def test_process_array_has_right_size(flat_market_3, flat_bvs_3):
    pa = build_process_array(flat_market_3, flat_bvs_3)
    assert pa.size() == 3
    assert pa.factors() == 3
    init = list(pa.initialValues())
    assert init == [100.0, 100.0, 100.0]


def test_surface_count_mismatch_rejected(flat_market_3, flat_bvs_3):
    with pytest.raises(ValueError):
        build_process_array(flat_market_3, flat_bvs_3[:2])
