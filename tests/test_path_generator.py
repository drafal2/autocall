"""Tests for the Sobol + Brownian-bridge path generator."""

import math

import numpy as np
import pytest
import QuantLib as ql

from autocall.market_data import MarketData
from autocall.path_generator import PathGenerator
from autocall.process import build_process_array
from autocall.time_grid import build_time_grid
from autocall.vol.implied_surface import build_black_variance_surface, flat_vol_matrix


def _single_asset_setup(eval_date, vol=0.20, T_years=1):
    md = MarketData.flat(
        eval_date=eval_date,
        spots=[100.0],
        risk_free=0.02,
        dividends=[0.0],
        correlation=np.array([[1.0]]),
    )
    strikes = [50.0, 100.0, 150.0]
    expiries = [eval_date + ql.Period(i, ql.Years) for i in (1, 2, 3)]
    bvs = [build_black_variance_surface(eval_date, flat_vol_matrix(strikes, expiries, vol))]
    pa = build_process_array(md, bvs)
    obs = [eval_date + ql.Period(T_years, ql.Years)]
    tg = build_time_grid(eval_date, obs, steps_per_year=24)
    return md, pa, tg


def test_european_call_matches_black_scholes(eval_date):
    md, pa, tg = _single_asset_setup(eval_date, vol=0.20, T_years=1)
    pg = PathGenerator(pa, tg, n_paths=2**14, seed=42, bridge=True)
    paths = pg.generate()
    K = 100.0
    T = 1.0
    disc = np.exp(-0.02 * T)
    payoff = np.maximum(paths[:, 0, -1] - K, 0.0) * disc
    mc_price = payoff.mean()
    se = payoff.std(ddof=1) / math.sqrt(len(payoff))

    # Analytic Black-Scholes via QuantLib
    spot = ql.QuoteHandle(ql.SimpleQuote(100.0))
    rf = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.02, ql.Actual365Fixed()))
    div = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, 0.0, ql.Actual365Fixed()))
    vol = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.TARGET(), 0.20, ql.Actual365Fixed())
    )
    process = ql.BlackScholesMertonProcess(spot, div, rf, vol)
    expiry = eval_date + ql.Period(1, ql.Years)
    exercise = ql.EuropeanExercise(expiry)
    payoff_ql = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.VanillaOption(payoff_ql, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    analytic = option.NPV()

    # MC should be within 4 standard errors of the analytic value.
    assert abs(mc_price - analytic) < 4 * se, (
        f"MC {mc_price:.4f} vs analytic {analytic:.4f} (SE={se:.4f})"
    )


def test_terminal_moments_match_geometric_bm(eval_date):
    md, pa, tg = _single_asset_setup(eval_date, vol=0.20, T_years=3)
    pg = PathGenerator(pa, tg, n_paths=2**13, seed=7, bridge=True)
    paths = pg.generate()
    log_ret = np.log(paths[:, 0, -1] / 100.0)
    # Risk-neutral analytics for flat vol GBM:
    T = 3.0
    sigma = 0.20
    mu_logret = (0.02 - 0.5 * sigma**2) * T
    sd_logret = sigma * math.sqrt(T)
    assert abs(log_ret.mean() - mu_logret) < 5 * sd_logret / math.sqrt(len(log_ret))
    assert abs(log_ret.std(ddof=1) - sd_logret) < 0.02


def test_correlation_recovered(eval_date, flat_market_3, flat_bvs_3):
    pa = build_process_array(flat_market_3, flat_bvs_3)
    obs = [eval_date + ql.Period(12, ql.Months)]
    tg = build_time_grid(eval_date, obs, steps_per_year=12)
    pg = PathGenerator(pa, tg, n_paths=4096, seed=42, bridge=True)
    paths = pg.generate()
    log_ret = np.diff(np.log(paths), axis=2)
    flat = log_ret.transpose(1, 0, 2).reshape(3, -1)
    emp = np.corrcoef(flat)
    # Off-diagonals = 0.4
    for i in range(3):
        for j in range(3):
            target = 1.0 if i == j else 0.4
            assert abs(emp[i, j] - target) < 0.03, (i, j, emp[i, j])


def test_seed_determinism(eval_date):
    md, pa, tg = _single_asset_setup(eval_date)
    p1 = PathGenerator(pa, tg, n_paths=256, seed=123, bridge=True).generate()
    p2 = PathGenerator(pa, tg, n_paths=256, seed=123, bridge=True).generate()
    np.testing.assert_array_equal(p1, p2)


def test_bridge_changes_paths_but_not_marginals(eval_date):
    """Bridge=True and bridge=False produce different paths but matching
    terminal moments (both are valid discretisations of the same SDE)."""
    md, pa, tg = _single_asset_setup(eval_date, T_years=2)
    n = 4096
    a = PathGenerator(pa, tg, n_paths=n, seed=99, bridge=False).generate()
    b = PathGenerator(pa, tg, n_paths=n, seed=99, bridge=True).generate()
    assert not np.allclose(a, b), "bridge should reshape paths"
    # Terminal means within 3 SE of each other
    se = a[:, 0, -1].std(ddof=1) / math.sqrt(n)
    assert abs(a[:, 0, -1].mean() - b[:, 0, -1].mean()) < 3 * se


def test_n_paths_must_be_positive(eval_date):
    md, pa, tg = _single_asset_setup(eval_date)
    with pytest.raises(ValueError):
        PathGenerator(pa, tg, n_paths=0)
