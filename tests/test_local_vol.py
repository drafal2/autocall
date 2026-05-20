"""Tests for the Dupire local-vol wrapper."""

import math

import QuantLib as ql

from autocall.vol.implied_surface import build_black_variance_surface, flat_vol_matrix
from autocall.vol.local_vol import build_local_vol_surface


def test_flat_black_vol_collapses_to_flat_local_vol(eval_date, flat_market_3):
    strikes = [50.0, 100.0, 150.0]
    expiries = [eval_date + ql.Period(i, ql.Years) for i in (1, 2, 3)]
    vol = 0.20
    bvs = build_black_variance_surface(
        eval_date, flat_vol_matrix(strikes, expiries, vol)
    )
    lv = build_local_vol_surface(
        bvs,
        flat_market_3.spots[0],
        flat_market_3.discount_curve,
        flat_market_3.dividend_curves[0],
    )

    # Dupire local vol of a flat Black surface == Black vol everywhere on the
    # interior of the calibration domain.
    for T in (0.25, 1.0, 2.5):
        for S in (80.0, 100.0, 120.0):
            assert math.isclose(lv.localVol(T, S, True), vol, abs_tol=1e-6), (
                f"failed at (T={T}, S={S}): got {lv.localVol(T, S, True)}"
            )
