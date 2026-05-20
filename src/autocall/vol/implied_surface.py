"""Build ``ql.BlackVarianceSurface`` from a (strike, expiry) implied-vol matrix."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import QuantLib as ql


def build_black_variance_surface(
    eval_date: ql.Date,
    vol_matrix: pd.DataFrame,
    calendar: ql.Calendar | None = None,
    day_counter: ql.DayCounter | None = None,
) -> ql.BlackVarianceSurface:
    """Build a Black variance surface from an implied vol matrix.

    Parameters
    ----------
    eval_date : ql.Date
        Reference date — index of the surface.
    vol_matrix : pd.DataFrame
        Index = strikes (float), columns = expiries (``ql.Date``).
        Values = Black implied vol (decimal, e.g. 0.20).
    calendar : ql.Calendar, optional
    day_counter : ql.DayCounter, optional

    Returns
    -------
    ql.BlackVarianceSurface with bilinear interpolation and extrapolation on.
    """
    cal = calendar or ql.TARGET()
    dc = day_counter or ql.Actual365Fixed()

    strikes: list[float] = [float(k) for k in vol_matrix.index]
    expiries: list[ql.Date] = list(vol_matrix.columns)
    for d in expiries:
        if not isinstance(d, ql.Date):
            raise TypeError(
                f"vol_matrix columns must be ql.Date objects, got {type(d).__name__}"
            )
        if d <= eval_date:
            raise ValueError(f"expiry {d} must be after eval_date {eval_date}")

    vol_qlmatrix = ql.Matrix(len(strikes), len(expiries))
    for i, _ in enumerate(strikes):
        for j, _ in enumerate(expiries):
            vol_qlmatrix[i][j] = float(vol_matrix.iloc[i, j])

    surface = ql.BlackVarianceSurface(
        eval_date, cal, expiries, strikes, vol_qlmatrix, dc
    )
    surface.enableExtrapolation()
    return surface


def build_handle(surface: ql.BlackVarianceSurface) -> ql.BlackVolTermStructureHandle:
    return ql.BlackVolTermStructureHandle(surface)


def flat_vol_matrix(
    strikes: Sequence[float],
    expiries: Sequence[ql.Date],
    vol: float,
) -> pd.DataFrame:
    """Convenience: produce a constant vol matrix for testing."""
    data = np.full((len(strikes), len(expiries)), float(vol))
    return pd.DataFrame(data, index=list(strikes), columns=list(expiries))
