"""Per-asset Local-Vol Black-Scholes processes assembled into a correlated array."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import QuantLib as ql

from .market_data import MarketData
from .vol.local_vol import build_local_vol_surface


def build_local_vol_process(
    spot: ql.SimpleQuote,
    risk_free: ql.YieldTermStructureHandle,
    dividend: ql.YieldTermStructureHandle,
    black_var_surface: ql.BlackVarianceSurface,
) -> ql.GeneralizedBlackScholesProcess:
    """Build a single-asset GBM process with Dupire local vol."""
    bvs_handle = ql.BlackVolTermStructureHandle(black_var_surface)
    lvs_handle = build_local_vol_surface(black_var_surface, spot, risk_free, dividend)
    return ql.GeneralizedBlackScholesProcess(
        ql.QuoteHandle(spot),
        dividend,
        risk_free,
        bvs_handle,
        lvs_handle,
    )


def build_process_array(
    market: MarketData,
    black_var_surfaces: Sequence[ql.BlackVarianceSurface],
) -> ql.StochasticProcessArray:
    """Wrap N per-asset local-vol processes into a correlated array.

    ``StochasticProcessArray`` performs the Cholesky decomposition of the
    correlation matrix internally; the caller does not need to pre-factorise.
    """
    if len(black_var_surfaces) != market.n_assets:
        raise ValueError(
            f"got {len(black_var_surfaces)} surfaces for {market.n_assets} assets"
        )

    processes = [
        build_local_vol_process(
            spot=market.spots[i],
            risk_free=market.discount_curve,
            dividend=market.dividend_curves[i],
            black_var_surface=black_var_surfaces[i],
        )
        for i in range(market.n_assets)
    ]

    corr = market.correlation
    ql_corr = ql.Matrix(corr.shape[0], corr.shape[1])
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ql_corr[i][j] = float(corr[i, j])

    return ql.StochasticProcessArray(processes, ql_corr)


def to_numpy_correlation(matrix: ql.Matrix) -> np.ndarray:
    n, m = matrix.rows(), matrix.columns()
    out = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = matrix[i][j]
    return out
