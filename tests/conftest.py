"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import QuantLib as ql

from autocall.market_data import MarketData
from autocall.vol.implied_surface import build_black_variance_surface, flat_vol_matrix


@pytest.fixture
def eval_date() -> ql.Date:
    d = ql.Date(15, 5, 2026)
    ql.Settings.instance().evaluationDate = d
    return d


@pytest.fixture
def flat_market_3(eval_date) -> MarketData:
    corr = np.array(
        [
            [1.0, 0.4, 0.4],
            [0.4, 1.0, 0.4],
            [0.4, 0.4, 1.0],
        ]
    )
    return MarketData.flat(
        eval_date=eval_date,
        spots=[100.0, 100.0, 100.0],
        risk_free=0.02,
        dividends=[0.0, 0.0, 0.0],
        correlation=corr,
    )


@pytest.fixture
def flat_bvs_3(eval_date):
    strikes = [50.0, 75.0, 100.0, 125.0, 150.0]
    expiries = [eval_date + ql.Period(i, ql.Years) for i in (1, 2, 3, 4)]
    return [
        build_black_variance_surface(eval_date, flat_vol_matrix(strikes, expiries, 0.20))
        for _ in range(3)
    ]
