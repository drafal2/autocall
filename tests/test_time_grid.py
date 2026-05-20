"""Tests for the TimeGrid builder."""

import pytest
import QuantLib as ql

from autocall.time_grid import build_time_grid, grid_indices_of_obs_dates


def test_obs_dates_are_mandatory_nodes(eval_date):
    obs = [eval_date + ql.Period(m, ql.Months) for m in (6, 12, 24, 36)]
    tg = build_time_grid(eval_date, obs, steps_per_year=24)
    idx = grid_indices_of_obs_dates(tg, eval_date, obs)
    assert len(idx) == len(obs)
    assert all(0 < i < len(tg) for i in idx)


def test_sub_steps_density(eval_date):
    obs = [eval_date + ql.Period(12, ql.Months)]
    tg = build_time_grid(eval_date, obs, steps_per_year=52)
    # ~52 weekly steps over 1Y
    assert 50 <= len(tg) - 1 <= 60


def test_unordered_obs_rejected(eval_date):
    obs = [
        eval_date + ql.Period(12, ql.Months),
        eval_date + ql.Period(6, ql.Months),
    ]
    with pytest.raises(ValueError, match="ascending"):
        build_time_grid(eval_date, obs, steps_per_year=12)


def test_obs_before_eval_rejected(eval_date):
    obs = [eval_date - 5]
    with pytest.raises(ValueError, match="after eval_date"):
        build_time_grid(eval_date, obs, steps_per_year=12)
