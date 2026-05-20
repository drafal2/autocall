"""Tests for the implied vol → Black variance surface builder."""

import math

import pytest
import QuantLib as ql

from autocall.vol.implied_surface import build_black_variance_surface, flat_vol_matrix


def test_flat_surface_round_trips(eval_date):
    strikes = [80.0, 100.0, 120.0]
    expiries = [eval_date + ql.Period(i, ql.Years) for i in (1, 2, 3)]
    vol = 0.20
    bvs = build_black_variance_surface(
        eval_date, flat_vol_matrix(strikes, expiries, vol)
    )
    # Query at nodes
    for K in strikes:
        for d in expiries:
            T = ql.Actual365Fixed().yearFraction(eval_date, d)
            assert math.isclose(
                bvs.blackVol(T, K), vol, abs_tol=1e-10
            ), f"node ({K}, {d})"
    # Query off-node
    T_mid = ql.Actual365Fixed().yearFraction(eval_date, expiries[0]) + 0.3
    for K in (90.0, 110.0):
        assert math.isclose(bvs.blackVol(T_mid, K), vol, abs_tol=1e-10)


def test_expiry_before_eval_rejected(eval_date):
    strikes = [100.0]
    expiries = [eval_date - 1]
    with pytest.raises(ValueError, match="after eval_date"):
        build_black_variance_surface(
            eval_date, flat_vol_matrix(strikes, expiries, 0.20)
        )
