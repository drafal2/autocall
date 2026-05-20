"""Tests for SVI fit."""

import math

import numpy as np
import pandas as pd
import pytest
import QuantLib as ql

from autocall.vol.svi import (
    SVIParams,
    fit_svi_slice,
    fit_svi_surface,
    svi_to_black_variance_surface,
)


def test_svi_recovers_known_params():
    true = SVIParams(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.1)
    k = np.linspace(-0.6, 0.6, 25)
    w = true.total_variance(k)
    fit = fit_svi_slice(k, w)
    # Recovered total variance should match closely on the fit grid.
    w_hat = fit.total_variance(k)
    assert np.allclose(w_hat, w, atol=1e-5), (w_hat - w)


def test_svi_surface_fits_synthetic_quotes(eval_date):
    F = 100.0
    expiries = [eval_date + ql.Period(i, ql.Years) for i in (1, 2, 3)]
    times = np.array([ql.Actual365Fixed().yearFraction(eval_date, d) for d in expiries])
    # Build synthetic vols from a smooth SVI per slice
    true_params = [
        SVIParams(a=0.01 * T, b=0.3 * T, rho=-0.4, m=0.0, sigma=0.15)
        for T in times
    ]
    strikes = np.linspace(70.0, 130.0, 11)
    k_grid = np.log(strikes / F)
    iv = np.zeros((len(strikes), len(expiries)))
    for j, (T, p) in enumerate(zip(times, true_params)):
        w = p.total_variance(k_grid)
        iv[:, j] = np.sqrt(w / T)
    quotes = pd.DataFrame(iv, index=list(strikes), columns=expiries)

    surf = fit_svi_surface(eval_date, quotes, forwards=[F] * 3)
    assert surf.check_calendar_arbitrage()  # synthetic data must be arb-free
    # The resampled BVS should roughly reproduce the input vols at the quote
    # strikes.
    bvs = svi_to_black_variance_surface(eval_date, surf, strike_grid=list(strikes))
    for j, T in enumerate(times):
        for i, K in enumerate(strikes):
            assert math.isclose(bvs.blackVol(T, K), iv[i, j], rel_tol=5e-3), (
                f"mismatch at ({K}, T={T})"
            )


def test_svi_rejects_non_positive_variance():
    with pytest.raises(ValueError):
        fit_svi_slice(np.array([-0.1, 0.0, 0.1]), np.array([0.04, 0.0, 0.04]))
