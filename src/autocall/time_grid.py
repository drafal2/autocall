"""Build a ``ql.TimeGrid`` that respects mandatory observation dates."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import QuantLib as ql


def build_time_grid(
    eval_date: ql.Date,
    obs_dates: Sequence[ql.Date],
    steps_per_year: int,
    day_counter: ql.DayCounter | None = None,
) -> ql.TimeGrid:
    """Build a TimeGrid containing all observation dates plus sub-steps.

    The grid is constructed so that every obs date appears as a mandatory node.
    Between two consecutive obs dates, the grid is uniformly refined to
    approximately ``steps_per_year`` sub-steps per year.

    The fine resolution is needed for:
      * accurate local-vol diffusion (Euler error scales with dt),
      * future American KI handling, which checks the barrier on each
        sub-step and applies a Brownian-bridge breach correction.
    """
    dc = day_counter or ql.Actual365Fixed()
    times: list[float] = [dc.yearFraction(eval_date, d) for d in obs_dates]
    if any(t <= 0 for t in times):
        raise ValueError("observation dates must be strictly after eval_date")
    if list(times) != sorted(times):
        raise ValueError("observation dates must be in ascending order")

    n_steps = max(1, int(np.ceil(times[-1] * steps_per_year)))
    return ql.TimeGrid(times, n_steps)


def grid_indices_of_obs_dates(
    time_grid: ql.TimeGrid,
    eval_date: ql.Date,
    obs_dates: Sequence[ql.Date],
    day_counter: ql.DayCounter | None = None,
    tol: float = 1e-8,
) -> list[int]:
    """Return the indices in ``time_grid`` corresponding to each obs date."""
    dc = day_counter or ql.Actual365Fixed()
    grid_times = np.array([time_grid[i] for i in range(len(time_grid))])
    out: list[int] = []
    for d in obs_dates:
        t = dc.yearFraction(eval_date, d)
        idx = int(np.argmin(np.abs(grid_times - t)))
        if abs(grid_times[idx] - t) > tol:
            raise ValueError(
                f"obs date {d} (t={t:.6f}) not found on grid; nearest "
                f"node = {grid_times[idx]:.6f}"
            )
        out.append(idx)
    return out
