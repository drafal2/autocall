"""Market-data container for the multi-asset diffusion engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import QuantLib as ql


def _validate_psd(matrix: np.ndarray, tol: float = 1e-10) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"correlation must be square 2-D, got shape {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=tol):
        raise ValueError("correlation matrix is not symmetric")
    diag = np.diag(matrix)
    if not np.allclose(diag, 1.0, atol=tol):
        raise ValueError("correlation matrix must have unit diagonal")
    eigvals = np.linalg.eigvalsh(matrix)
    if eigvals.min() < -tol:
        raise ValueError(
            f"correlation matrix is not PSD (min eigenvalue = {eigvals.min():.3e}). "
            "TODO: nearest-PSD repair."
        )


@dataclass
class MarketData:
    """N-asset market data snapshot.

    Spots are stored as ``ql.SimpleQuote`` so they can be bumped in-place for
    finite-difference Greeks in a later iteration.
    """

    eval_date: ql.Date
    spots: list[ql.SimpleQuote]
    discount_curve: ql.YieldTermStructureHandle
    dividend_curves: list[ql.YieldTermStructureHandle]
    correlation: np.ndarray
    calendar: ql.Calendar = field(default_factory=ql.TARGET)
    day_counter: ql.DayCounter = field(default_factory=ql.Actual365Fixed)

    def __post_init__(self) -> None:
        n = len(self.spots)
        if len(self.dividend_curves) != n:
            raise ValueError(
                f"dividend_curves length {len(self.dividend_curves)} != n_assets {n}"
            )
        if self.correlation.shape != (n, n):
            raise ValueError(
                f"correlation shape {self.correlation.shape} != ({n}, {n})"
            )
        _validate_psd(self.correlation)
        ql.Settings.instance().evaluationDate = self.eval_date

    @property
    def n_assets(self) -> int:
        return len(self.spots)

    @property
    def spot_handles(self) -> list[ql.QuoteHandle]:
        return [ql.QuoteHandle(s) for s in self.spots]

    @classmethod
    def flat(
        cls,
        eval_date: ql.Date,
        spots: Sequence[float],
        risk_free: float,
        dividends: Sequence[float],
        correlation: np.ndarray,
        calendar: ql.Calendar | None = None,
        day_counter: ql.DayCounter | None = None,
    ) -> "MarketData":
        """Convenience constructor for flat curves (testing / smoke runs)."""
        cal = calendar or ql.TARGET()
        dc = day_counter or ql.Actual365Fixed()
        rf_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(eval_date, risk_free, dc)
        )
        div_handles = [
            ql.YieldTermStructureHandle(ql.FlatForward(eval_date, q, dc))
            for q in dividends
        ]
        spot_quotes = [ql.SimpleQuote(s) for s in spots]
        return cls(
            eval_date=eval_date,
            spots=spot_quotes,
            discount_curve=rf_handle,
            dividend_curves=div_handles,
            correlation=np.asarray(correlation, dtype=float),
            calendar=cal,
            day_counter=dc,
        )
