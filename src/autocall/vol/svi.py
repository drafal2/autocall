"""Raw SVI per-slice fit and resampling to a Black variance surface.

Raw SVI (Gatheral 2004):

    w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )

where ``k = log(K / F)`` is log-moneyness and ``w`` is total implied variance
(i.e. ``sigma_IV^2 * T``).

Per-slice fit constraints (no-static-arbitrage on a single slice):
    b >= 0,  |rho| < 1,  sigma > 0,  a + b * sigma * sqrt(1 - rho^2) >= 0.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.optimize import minimize

from .implied_surface import build_black_variance_surface


@dataclass
class SVIParams:
    """Raw SVI parameters for one expiry slice."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray | float) -> np.ndarray | float:
        k_arr = np.asarray(k, dtype=float)
        return self.a + self.b * (
            self.rho * (k_arr - self.m)
            + np.sqrt((k_arr - self.m) ** 2 + self.sigma**2)
        )


def fit_svi_slice(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    weights: np.ndarray | None = None,
) -> SVIParams:
    """Fit raw SVI to a single expiry slice in total-variance space.

    The loss is weighted MSE on **implied vol** (sqrt(w / T) requires T,
    but since this is one slice, sqrt(w) suffices up to a constant and is
    more numerically stable than fitting on w directly).
    """
    k = np.asarray(log_moneyness, dtype=float)
    w_mkt = np.asarray(total_variance, dtype=float)
    if w_mkt.min() <= 0:
        raise ValueError("total variance must be strictly positive")
    if weights is None:
        weights = np.ones_like(k)
    weights = np.asarray(weights, dtype=float)

    iv_mkt = np.sqrt(w_mkt)

    # Initial guess: a at min variance, b small slope, rho moderate skew,
    # m at the strike with minimum variance, sigma at a typical curvature.
    a0 = max(float(w_mkt.min()) * 0.5, 1e-6)
    b0 = 0.1
    rho0 = -0.3
    m0 = float(k[np.argmin(w_mkt)])
    sigma0 = 0.1
    x0 = np.array([a0, b0, rho0, m0, sigma0])

    def objective(x: np.ndarray) -> float:
        a, b, rho, m, sigma = x
        w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        if np.any(w_model <= 0):
            return 1e6
        iv_model = np.sqrt(w_model)
        return float(np.sum(weights * (iv_model - iv_mkt) ** 2))

    bounds = [
        (-1.0, 5.0),     # a
        (0.0, 5.0),      # b >= 0
        (-0.999, 0.999), # |rho| < 1
        (-2.0, 2.0),     # m
        (1e-4, 5.0),     # sigma > 0
    ]

    # Linear no-arb floor: a + b*sigma*sqrt(1-rho^2) >= 0.
    def arb_floor(x: np.ndarray) -> float:
        a, b, _, _, sigma = x
        rho = x[2]
        return a + b * sigma * math.sqrt(max(1.0 - rho * rho, 0.0))

    constraints = [{"type": "ineq", "fun": arb_floor}]

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not res.success:
        warnings.warn(f"SVI fit did not fully converge: {res.message}")

    return SVIParams(a=res.x[0], b=res.x[1], rho=res.x[2], m=res.x[3], sigma=res.x[4])


@dataclass
class SVISurface:
    """Collection of per-expiry SVI slices."""

    expiries: list[ql.Date]
    times: np.ndarray          # year fractions
    params: list[SVIParams]
    forwards: np.ndarray       # F(T_i) used to define log-moneyness

    def total_variance(self, k: float, t_index: int) -> float:
        return float(self.params[t_index].total_variance(k))

    def implied_vol(self, strike: float, t_index: int) -> float:
        F = self.forwards[t_index]
        k = math.log(strike / F)
        w = self.total_variance(k, t_index)
        T = self.times[t_index]
        return math.sqrt(max(w / T, 1e-12))

    def check_calendar_arbitrage(
        self,
        k_grid: np.ndarray | None = None,
        tol: float = 1e-8,
    ) -> bool:
        """Return True if total variance is non-decreasing in T at every k.

        Calendar arbitrage exists iff w(k, T_{i+1}) < w(k, T_i) for some k.
        Emits a warning if violated and returns False.
        """
        if k_grid is None:
            k_grid = np.linspace(-1.0, 1.0, 51)
        ok = True
        for i in range(len(self.params) - 1):
            w_i = self.params[i].total_variance(k_grid)
            w_next = self.params[i + 1].total_variance(k_grid)
            if np.any(w_next < w_i - tol):
                warnings.warn(
                    f"Calendar arbitrage between expiry {i} ({self.expiries[i]}) "
                    f"and {i + 1} ({self.expiries[i + 1]})."
                )
                ok = False
        return ok


def fit_svi_surface(
    eval_date: ql.Date,
    quotes: pd.DataFrame,
    forwards: Sequence[float],
    day_counter: ql.DayCounter | None = None,
) -> SVISurface:
    """Fit one SVI slice per expiry.

    Parameters
    ----------
    eval_date : ql.Date
    quotes : pd.DataFrame
        Index = strikes, columns = expiries (``ql.Date``), values = implied vol.
    forwards : sequence of float
        Forward price per expiry, same order as ``quotes.columns``.
    day_counter : ql.DayCounter, optional
    """
    dc = day_counter or ql.Actual365Fixed()
    strikes = np.array([float(k) for k in quotes.index])
    expiries = list(quotes.columns)
    if len(forwards) != len(expiries):
        raise ValueError("len(forwards) must match number of expiries")

    times = np.array([dc.yearFraction(eval_date, d) for d in expiries])
    params: list[SVIParams] = []
    for j, (T, F) in enumerate(zip(times, forwards)):
        iv = quotes.iloc[:, j].to_numpy(dtype=float)
        w = iv**2 * T
        k = np.log(strikes / F)
        params.append(fit_svi_slice(k, w))

    return SVISurface(
        expiries=expiries,
        times=times,
        params=params,
        forwards=np.asarray(forwards, dtype=float),
    )


def svi_to_black_variance_surface(
    eval_date: ql.Date,
    svi: SVISurface,
    strike_grid: Sequence[float],
    calendar: ql.Calendar | None = None,
    day_counter: ql.DayCounter | None = None,
) -> ql.BlackVarianceSurface:
    """Resample the fitted SVI on a strike x expiry grid and wrap in BVS."""
    cal = calendar or ql.TARGET()
    dc = day_counter or ql.Actual365Fixed()
    strikes = list(strike_grid)
    vol_data = np.zeros((len(strikes), len(svi.expiries)))
    for j, F in enumerate(svi.forwards):
        for i, K in enumerate(strikes):
            vol_data[i, j] = svi.implied_vol(float(K), j)
    df = pd.DataFrame(vol_data, index=strikes, columns=svi.expiries)
    return build_black_variance_surface(eval_date, df, calendar=cal, day_counter=dc)
