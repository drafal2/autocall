"""Tests for MarketData."""

import numpy as np
import pytest
import QuantLib as ql

from autocall.market_data import MarketData


def test_flat_constructs(eval_date):
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    md = MarketData.flat(
        eval_date=eval_date,
        spots=[100.0, 120.0],
        risk_free=0.02,
        dividends=[0.0, 0.01],
        correlation=corr,
    )
    assert md.n_assets == 2
    assert md.spots[0].value() == 100.0
    assert md.spots[1].value() == 120.0
    assert len(md.dividend_curves) == 2


def test_non_unit_diagonal_rejected(eval_date):
    corr = np.array([[1.0, 0.5], [0.5, 0.9]])
    with pytest.raises(ValueError, match="unit diagonal"):
        MarketData.flat(eval_date, [100.0, 100.0], 0.02, [0.0, 0.0], corr)


def test_non_symmetric_rejected(eval_date):
    corr = np.array([[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        MarketData.flat(eval_date, [100.0, 100.0], 0.02, [0.0, 0.0], corr)


def test_non_psd_rejected(eval_date):
    corr = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="not PSD"):
        MarketData.flat(eval_date, [100.0] * 3, 0.02, [0.0] * 3, corr)


def test_size_mismatch_rejected(eval_date):
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    with pytest.raises(ValueError):
        MarketData.flat(eval_date, [100.0, 100.0, 100.0], 0.02, [0.0] * 3, corr)
