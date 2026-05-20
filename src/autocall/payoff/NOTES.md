# Payoff scope (next PR)

The diffusion engine in this PR is payoff-agnostic. The payoff layer will live in
`autocall.payoff` and consume `PathGenerator.generate()` output of shape
`(n_paths, n_assets, n_steps + 1)`.

## Product types to support

### Phoenix with memory
- Worst-of basket performance evaluated on each observation date.
- Autocall: if `WO_i >= autocall_barrier_i`, redeem with `nominal + Σ unpaid coupons`.
- Conditional coupon: paid on obs date `i` if `WO_i >= coupon_barrier_i`.
  Memory feature: if `WO_i >= coupon_barrier_i`, ALSO pay any previously
  unpaid coupons (catch-up).
- Final maturity (no prior autocall):
  - `WO_T >= KI_barrier` → `nominal + final coupon` (if cond met).
  - `WO_T <  KI_barrier` → `nominal * WO_T` (downside exposure).

### Athena
- No conditional coupons during the life.
- On autocall date `i`: pay `nominal * (1 + i * C)` where `C` is the per-period
  coupon rate.
- At maturity (no autocall): `nominal` if `WO_T >= KI`, else `nominal * WO_T`.

### Snowball
- Coupons compound: pays `nominal * (1 + C)^i` on autocall date `i`.
- KI logic identical to Phoenix/Athena at maturity.

## KI variants

### European KI (default, cheap)
- Check `WO_T < KI` only on the final date.
- Only the obs-date slice of the path is consumed; sub-steps unused for KI.

### American (continuous) KI
- KI fires if `WO_t < KI` at any t during the life.
- Use the existing fine time grid from `time_grid.py` (sub-steps per obs interval).
- Apply Brownian-bridge breach-probability correction per sub-interval to
  reduce discretization bias:
  for each sub-step `[t_k, t_{k+1}]` with end-points `S_k, S_{k+1}` both above
  the barrier `B`, the conditional probability of breaching is
  `exp( -2 * ln(S_k / B) * ln(S_{k+1} / B) / (sigma^2 * dt) )`.
  Sample a uniform per sub-step to decide breach.

## Public API sketch

```python
from autocall.payoff import PhoenixPayoff, AthenaPayoff, SnowballPayoff, KIStyle

payoff = PhoenixPayoff(
    notional=1_000_000,
    obs_dates=obs_dates,
    autocall_barriers=[1.0] * len(obs_dates),
    coupon_barriers=[0.7] * len(obs_dates),
    coupon_rate=0.02,
    memory=True,
    ki_barrier=0.6,
    ki_style=KIStyle.AMERICAN,
)

price, se = engine.price(payoff, market_data, n_paths=2**15)
```
