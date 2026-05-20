"""Dupire local volatility wrapper per asset."""

from __future__ import annotations

import QuantLib as ql


def build_local_vol_surface(
    black_var_surface: ql.BlackVarianceSurface,
    spot: ql.SimpleQuote,
    risk_free: ql.YieldTermStructureHandle,
    dividend: ql.YieldTermStructureHandle,
) -> ql.LocalVolTermStructureHandle:
    """Build a Dupire local-vol term structure handle for one asset.

    Notes
    -----
    ``ql.LocalVolSurface`` derives local vol from the Black variance surface
    via the Dupire formula. ``NoExceptLocalVolSurface`` is wrapped on top
    so that arbitrage holes (negative dupire numerator) clamp to a small
    positive vol instead of raising, which is essential for path generation
    near the boundaries of the calibration grid.
    """
    bvs_handle = ql.BlackVolTermStructureHandle(black_var_surface)
    spot_handle = ql.QuoteHandle(spot)

    safe = ql.NoExceptLocalVolSurface(
        bvs_handle,
        risk_free,
        dividend,
        spot_handle,
        1e-4,  # floor (vol, not variance) when Dupire is undefined
    )
    safe.enableExtrapolation()
    return ql.LocalVolTermStructureHandle(safe)
