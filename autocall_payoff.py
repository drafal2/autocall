"""Payoff and pricing engine for multi-asset equity autocall notes.

The engine prices an autocall note on *pre-simulated* underlying paths (e.g. the
``GaussianSobolMultiPathGenerator`` output produced in ``notebooks/autocall.ipynb``)
and keeps a fully auditable, per-path cashflow record.

Conventions
-----------
* Performance of constituent ``i`` at observation ``t`` is ``S_i(t) / S_i(0)``,
  where ``S_i(0)`` are the supplied initial fixings.
* No notional / principal is exchanged: the holder only ever receives interest
  (early-redemption coupon) and guaranteed coupons. A path that is never
  redeemed pays ``0`` at maturity.
* The autocall barrier is fixed across observation dates, and for memory-on-KO
  the KO level equals that same barrier.

Two timing conventions are not pinned down by the term sheet and are made
explicit here (change in one place if your desk defines them differently):

1. ``early_redemption_rate`` and ``guaranteed_coupon`` are **per-period** amounts.
   Snowball / digital accumulation multiplies by the number of elapsed periods
   ``k`` (1-based count of observation dates reached).
2. A **digital** guaranteed coupon pays a single lump ``c * k`` on the path's
   termination date (the redemption date, or maturity if never redeemed).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import QuantLib as ql


class BasketMode(str, Enum):
    """How individual constituent performances are aggregated into a basket."""

    BEST_OF = "best_of"            # metric = max_i(perf); redeem when metric <= barrier
    WORST_OF = "worst_of"          # metric = min_i(perf); redeem when metric >= barrier
    MULTIPERFORMANCE = "multiperformance"  # metric = sum_i w_i*perf_i; redeem when >= barrier


class RedemptionType(str, Enum):
    """Early-redemption mechanism."""

    USUAL = "usual"          # basket metric vs fixed barrier on each date
    MEMORY_KO = "memory_ko"  # redeem once every constituent has breached at least once


@dataclass
class AutocallTerms:
    """Product definition for an autocall note.

    Parameters
    ----------
    autocall_barrier : float
        Fixed barrier expressed in performance terms (e.g. ``1.0`` for 100%).
        Also used as the KO level for :attr:`RedemptionType.MEMORY_KO`.
    redemption_type : RedemptionType
        Early-redemption mechanism.
    basket_mode : BasketMode
        Basket aggregation / breach direction.
    weights : sequence of float, optional
        Constituent weights, required for ``MULTIPERFORMANCE``.
    early_redemption_rate : float
        Per-period interest paid on early redemption.
    snowball : bool
        If ``True`` the early-redemption interest accumulates as ``rate * k``.
    guaranteed_coupon : float
        Per-period guaranteed coupon paid while the note is alive.
    coupon_digital : bool
        If ``True`` the guaranteed coupon is paid as a single lump ``c * k`` on
        the termination date; otherwise ``c`` is paid on each surviving date.
    coupon_on_redemption : bool
        If ``True`` the guaranteed coupon for the redemption period is also paid
        on the redemption date.
    notional : float
        Position size used to convert the per-unit price into a market value
        (``market_value = notional * price``) and to express cashflows in
        notional terms. It does **not** affect the per-unit price itself, and
        no principal is exchanged.
    """

    autocall_barrier: float
    redemption_type: RedemptionType = RedemptionType.USUAL
    basket_mode: BasketMode = BasketMode.WORST_OF
    weights: Optional[Sequence[float]] = None

    early_redemption_rate: float = 0.0
    snowball: bool = False

    guaranteed_coupon: float = 0.0
    coupon_digital: bool = False
    coupon_on_redemption: bool = False

    notional: float = 1.0

    def __post_init__(self) -> None:
        self.redemption_type = RedemptionType(self.redemption_type)
        self.basket_mode = BasketMode(self.basket_mode)

        if self.basket_mode is BasketMode.MULTIPERFORMANCE:
            if self.weights is None:
                raise ValueError("weights are required for a multiperformance basket")
            if self.redemption_type is RedemptionType.MEMORY_KO:
                raise ValueError(
                    "memory-on-KO is per-constituent and is not defined for a "
                    "multiperformance basket"
                )
        elif self.weights is not None:
            raise ValueError("weights only apply to a multiperformance basket")


def _ql_to_pydate(d: ql.Date) -> datetime.date:
    """Convert a ``ql.Date`` to a ``datetime.date`` for tidy DataFrame output."""
    return datetime.date(d.year(), d.month(), d.dayOfMonth())


class AutocallNote:
    """Price an autocall note on pre-simulated paths and retain an audit trail.

    Parameters
    ----------
    terms : AutocallTerms
        Product definition.
    initial_fixings : array-like, shape (n_assets,)
        Initial fixing level ``S_i(0)`` of each constituent.
    paths : ndarray, shape (n_paths, n_assets, n_cols)
        Pre-simulated spot paths. Column ``0`` is the valuation date.
    path_dates : sequence of ql.Date, length n_cols
        Calendar date attached to each path column (including the valuation date).
    schedule : sequence of ql.Date
        Observation / payment dates. Must be a subset of ``path_dates`` and must
        not coincide with the valuation-date column.
    discount_curve : ql.YieldTermStructureHandle
        Curve used to discount cashflows via ``discount_curve.discount(date)``.

    Attributes
    ----------
    audit : pandas.DataFrame
        Per-``(path, obs)`` cashflow record, populated after :meth:`price`.
    price_ : float
        Monte Carlo price per unit notional (mean of per-path discounted
        cashflows), i.e. a fraction / percentage of notional.
    stderr_ : float
        Monte Carlo standard error of :attr:`price_`.
    market_value_ : float
        Market value of the position, ``notional * price_``.
    mv_stderr_ : float
        Monte Carlo standard error of :attr:`market_value_`.
    path_pv_ : ndarray, shape (n_paths,)
        Discounted per-unit cashflow sum for each path.
    """

    def __init__(
        self,
        terms: AutocallTerms,
        initial_fixings: Sequence[float],
        paths: np.ndarray,
        path_dates: Sequence[ql.Date],
        schedule: Sequence[ql.Date],
        discount_curve: ql.YieldTermStructureHandle,
    ) -> None:
        self.terms = terms
        self.initial_fixings = np.asarray(initial_fixings, dtype=float)
        self.paths = np.asarray(paths, dtype=float)
        self.path_dates = list(path_dates)
        self.schedule = list(schedule)
        self.discount_curve = discount_curve

        self.audit: Optional[pd.DataFrame] = None
        self.price_: Optional[float] = None
        self.stderr_: Optional[float] = None
        self.market_value_: Optional[float] = None
        self.mv_stderr_: Optional[float] = None
        self.path_pv_: Optional[np.ndarray] = None

        self._validate()

    # ------------------------------------------------------------------ setup
    def _validate(self) -> None:
        if self.paths.ndim != 3:
            raise ValueError(
                f"paths must be 3-D (n_paths, n_assets, n_cols); got {self.paths.shape}"
            )
        n_paths, n_assets, n_cols = self.paths.shape

        if self.initial_fixings.shape != (n_assets,):
            raise ValueError(
                f"initial_fixings must have shape ({n_assets},); "
                f"got {self.initial_fixings.shape}"
            )
        if len(self.path_dates) != n_cols:
            raise ValueError(
                f"path_dates length {len(self.path_dates)} != number of path "
                f"columns {n_cols}"
            )

        # Map each schedule date to its path column via the QuantLib serial number
        # (ql.Date is not reliably hashable across builds).
        date_to_col = {d.serialNumber(): i for i, d in enumerate(self.path_dates)}
        self.obs_cols = []
        missing = []
        for d in self.schedule:
            col = date_to_col.get(d.serialNumber())
            if col is None:
                missing.append(_ql_to_pydate(d))
            else:
                self.obs_cols.append(col)
        if missing:
            raise ValueError(
                f"schedule dates not present in path_dates: {missing}"
            )
        if any(c == 0 for c in self.obs_cols):
            raise ValueError("an observation date coincides with the valuation-date column")

        if self.terms.basket_mode is BasketMode.MULTIPERFORMANCE:
            w = np.asarray(self.terms.weights, dtype=float)
            if w.shape != (n_assets,):
                raise ValueError(
                    f"weights must have shape ({n_assets},); got {w.shape}"
                )
            self._weights = w
        else:
            self._weights = None

    # ------------------------------------------------------------- mechanics
    def _basket_metric(self, perf: np.ndarray) -> np.ndarray:
        """Aggregate constituent performances ``(n_paths, n_assets)`` to a basket."""
        mode = self.terms.basket_mode
        if mode is BasketMode.BEST_OF:
            return perf.max(axis=1)
        if mode is BasketMode.WORST_OF:
            return perf.min(axis=1)
        return perf @ self._weights  # multiperformance

    def _usual_trigger(self, metric: np.ndarray) -> np.ndarray:
        """Boolean redemption trigger for the 'usual' mechanism."""
        barrier = self.terms.autocall_barrier
        if self.terms.basket_mode is BasketMode.BEST_OF:
            return metric <= barrier            # redeem when the best falls below
        return metric >= barrier                # worst-of / multiperformance: above

    def _ko_breach(self, perf: np.ndarray) -> np.ndarray:
        """Per-constituent breach flags ``(n_paths, n_assets)`` for memory-on-KO."""
        barrier = self.terms.autocall_barrier
        if self.terms.basket_mode is BasketMode.BEST_OF:
            return perf <= barrier
        return perf >= barrier                  # worst-of

    # --------------------------------------------------------------- pricing
    def price(self) -> float:
        """Run the Monte Carlo valuation and populate the audit trail."""
        n_paths, n_assets, _ = self.paths.shape
        n_obs = len(self.obs_cols)
        t = self.terms

        alive = np.ones(n_paths, dtype=bool)
        breached_ever = np.zeros((n_paths, n_assets), dtype=bool)

        perf_store = np.zeros((n_paths, n_obs, n_assets))
        metric_store = np.zeros((n_paths, n_obs))
        redeemed_store = np.zeros((n_paths, n_obs), dtype=bool)
        coupon_store = np.zeros((n_paths, n_obs))
        interest_store = np.zeros((n_paths, n_obs))

        redeem_idx = np.full(n_paths, -1, dtype=int)
        redeemed_any = np.zeros(n_paths, dtype=bool)

        for j, col in enumerate(self.obs_cols):
            k = j + 1  # 1-based elapsed period count
            perf = self.paths[:, :, col] / self.initial_fixings
            perf_store[:, j, :] = perf

            if t.redemption_type is RedemptionType.USUAL:
                metric = self._basket_metric(perf)
                metric_store[:, j] = metric
                trigger = self._usual_trigger(metric)
            else:  # memory-on-KO
                breached_ever |= self._ko_breach(perf)
                metric_store[:, j] = breached_ever.sum(axis=1)  # # constituents breached
                trigger = breached_ever.all(axis=1)

            newly = alive & trigger
            redeemed_store[:, j] = newly
            redeem_idx[newly] = j
            redeemed_any |= newly

            # Early-redemption interest, per unit notional (no principal exchange)
            er = t.early_redemption_rate * k if t.snowball else t.early_redemption_rate
            interest_store[newly, j] = er

            # Guaranteed coupon -- per-period mode is paid here
            if not t.coupon_digital:
                pay_mask = alive & (~newly | t.coupon_on_redemption)
                coupon_store[pay_mask, j] = t.guaranteed_coupon

            alive &= ~newly

        # Termination period (1-based) and column for each path
        term_k = np.where(redeemed_any, redeem_idx + 1, n_obs)
        term_col = np.where(redeemed_any, redeem_idx, n_obs - 1)

        # Guaranteed coupon -- digital lump c*k paid once on the termination date
        if t.coupon_digital:
            credited = term_k.copy()
            if not t.coupon_on_redemption:
                credited = np.where(redeemed_any, term_k - 1, term_k)
            credited = np.clip(credited, 0, None)
            coupon_store[np.arange(n_paths), term_col] += (
                t.guaranteed_coupon * credited
            )

        cashflow_store = coupon_store + interest_store
        dfs = np.array(
            [self.discount_curve.discount(self.path_dates[c]) for c in self.obs_cols]
        )
        discounted = cashflow_store * dfs[None, :]

        self.path_pv_ = discounted.sum(axis=1)
        self.price_ = float(self.path_pv_.mean())
        self.stderr_ = float(self.path_pv_.std(ddof=1) / np.sqrt(n_paths))

        # Market value scales the per-unit price by the notional position size.
        self.market_value_ = self.price_ * t.notional
        self.mv_stderr_ = self.stderr_ * t.notional

        self._build_audit(
            perf_store, metric_store, redeemed_store, coupon_store,
            interest_store, cashflow_store, discounted, dfs,
        )
        return self.price_

    # --------------------------------------------------------------- reports
    def _build_audit(
        self,
        perf, metric, redeemed, coupon, interest, cashflow, discounted, dfs,
    ) -> None:
        n_paths, n_obs, n_assets = perf.shape
        obs_dates = [_ql_to_pydate(self.path_dates[c]) for c in self.obs_cols]

        index = pd.MultiIndex.from_product(
            [range(n_paths), range(n_obs)], names=["path", "obs"]
        )
        notional = self.terms.notional
        data = {f"perf_{a}": perf[:, :, a].reshape(-1) for a in range(n_assets)}
        data.update(
            basket_metric=metric.reshape(-1),
            redeemed=redeemed.reshape(-1),
            coupon=coupon.reshape(-1),
            redemption_interest=interest.reshape(-1),
            cashflow=cashflow.reshape(-1),
            discount_factor=np.broadcast_to(dfs, (n_paths, n_obs)).reshape(-1),
            discounted_cf=discounted.reshape(-1),
            cashflow_notional=(cashflow * notional).reshape(-1),
            discounted_cf_notional=(discounted * notional).reshape(-1),
        )
        df = pd.DataFrame(data, index=index)
        df.insert(0, "date", np.tile(obs_dates, n_paths))
        self.audit = df

    def expected_cashflows(self) -> pd.DataFrame:
        """Mean (undiscounted and discounted) cashflow per observation date."""
        if self.audit is None:
            raise RuntimeError("call price() first")
        grouped = self.audit.groupby("obs")
        out = grouped.agg(
            date=("date", "first"),
            redemption_prob=("redeemed", "mean"),
            mean_coupon=("coupon", "mean"),
            mean_interest=("redemption_interest", "mean"),
            mean_cashflow=("cashflow", "mean"),
            discount_factor=("discount_factor", "first"),
            mean_discounted_cf=("discounted_cf", "mean"),
            mean_cashflow_notional=("cashflow_notional", "mean"),
            mean_discounted_cf_notional=("discounted_cf_notional", "mean"),
        )
        return out

    def summary(self) -> dict:
        """Headline pricing results."""
        if self.price_ is None:
            raise RuntimeError("call price() first")
        return {
            "price": self.price_,
            "stderr": self.stderr_,
            "notional": self.terms.notional,
            "market_value": self.market_value_,
            "mv_stderr": self.mv_stderr_,
            "n_paths": self.paths.shape[0],
            "redemption_prob_total": float(
                self.audit.groupby("path")["redeemed"].any().mean()
            ),
        }
