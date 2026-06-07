"""
Multi-underlying autocall pricing with Sobol + Brownian bridge.

WHY THIS IS STRUCTURED THE WAY IT IS
------------------------------------
QuantLib's multi-asset path generator (MultiPathGenerator / GaussianSobolMultiPathGenerator)
does NOT implement the Brownian bridge: with brownianBridge=True it raises
"Brownian bridge not supported". Only the 1-D PathGenerator bridges. The lower-level
SobolBrownianGenerator that *is* built for the multi-factor case is not reliably exposed
in the Python bindings.

So we use QuantLib for what it exports cleanly and correctly:
    - ql.SobolRsg               : low-discrepancy uniforms (good direction numbers)
    - ql.InverseCumulativeNormal: uniform -> N(0,1) by inversion (NOT Box-Muller;
                                  Box-Muller destroys the low-discrepancy structure)
and supply our own thin Brownian-bridge layer that does the cross-factor Sobol
dimension allocation, which is exactly where the QMC speed-up is won or lost.

IMPORTANT - I could not execute this (QuantLib not available offline). Treat it as a
draft to validate. The function verify_bridge_covariance() is the acceptance test:
run it first. If it passes, the bridge math and scaling are correct.

Dimension layout (the crux for QMC):
    D = d_assets * m_dates Sobol dimensions per path.
    We reshape the D normals to (m, d) so that bridge-step s, factor f reads Sobol
    dimension s*d + f. Bridge step 0 is the TERMINAL of every factor, so Sobol dims
    0..d-1 carry all d terminal moves (the highest-variance directions), dims d..2d-1
    carry the first bisection, etc. That interleaving is what forward construction
    and a naive per-factor contiguous layout both get wrong.
"""

import numpy as np

try:
    import QuantLib as ql
    _HAVE_QL = True
except ImportError:
    _HAVE_QL = False


# ---------------------------------------------------------------------------
# 1. Brownian bridge (Glasserman, "Monte Carlo Methods in Financial Engineering")
#    Vectorised across paths. Produces path LEVELS W(t_k), correctly time-scaled
#    so Var(W(t_k)) = t_k and Cov(W(s),W(t)) = min(s,t). Scaling is explicit here
#    precisely so we do not depend on QuantLib's (ambiguous) scaling convention.
# ---------------------------------------------------------------------------
class BrownianBridge:
    def __init__(self, times):
        times = np.asarray(times, dtype=float)
        assert np.all(np.diff(times) > 0) and times[0] > 0, "times must be strictly increasing and > 0"
        n = len(times)
        self.n = n
        t_full = np.concatenate(([0.0], times))   # t_full[0] = 0

        self.bridge_index = np.empty(n, dtype=int)
        self.left_index = np.zeros(n, dtype=int)
        self.right_index = np.zeros(n, dtype=int)
        self.left_weight = np.zeros(n)
        self.right_weight = np.zeros(n)
        self.std_dev = np.zeros(n)

        mapped = np.zeros(n, dtype=int)
        self.bridge_index[0] = n - 1
        self.std_dev[0] = np.sqrt(times[n - 1])    # terminal: std = sqrt(T)
        mapped[n - 1] = 1

        j = 0
        for i in range(1, n):
            while mapped[j]:
                j += 1
            k = j
            while not mapped[k]:
                k += 1
            l = j + ((k - 1 - j) >> 1)             # midpoint to generate
            mapped[l] = 1
            self.bridge_index[i] = l
            self.left_index[i] = j
            self.right_index[i] = k
            t_l = t_full[j]                        # left anchor time (0 if j==0)
            t_i = t_full[l + 1]
            t_r = t_full[k + 1]
            self.left_weight[i] = (t_r - t_i) / (t_r - t_l)
            self.right_weight[i] = (t_i - t_l) / (t_r - t_l)
            self.std_dev[i] = np.sqrt((t_i - t_l) * (t_r - t_i) / (t_r - t_l))
            j = k + 1
            if j >= n:
                j = 0

    def transform(self, z):
        """z: (N, n) iid N(0,1) in BRIDGE order. Returns (N, n) path levels in TIME order."""
        z = np.atleast_2d(z)
        N = z.shape[0]
        path = np.empty((N, self.n))
        path[:, self.bridge_index[0]] = self.std_dev[0] * z[:, 0]
        for i in range(1, self.n):
            li, ri, bi = self.left_index[i], self.right_index[i], self.bridge_index[i]
            left_val = path[:, li - 1] if li != 0 else 0.0
            path[:, bi] = (self.left_weight[i] * left_val
                           + self.right_weight[i] * path[:, ri]
                           + self.std_dev[i] * z[:, i])
        return path


# ---------------------------------------------------------------------------
# 2. Sobol normals via QuantLib (with numpy fallback so the file runs without QL)
# ---------------------------------------------------------------------------
def sobol_normals(n_paths, dim, seed=42, skip=1024):
    """Return (n_paths, dim) standard normals from a Sobol sequence by inversion."""
    if _HAVE_QL:
        rsg = ql.SobolRsg(dim, seed, ql.SobolRsg.JoeKuoD7)   # high-dim direction numbers
        inv = ql.InverseCumulativeNormal()
        for _ in range(skip):                                # burn-in
            rsg.nextSequence()
        out = np.empty((n_paths, dim))
        for p in range(n_paths):
            u = rsg.nextSequence().value()
            out[p, :] = [inv(x) for x in u]
        return out
    # Fallback: pseudo-random (lets the verification test run without QuantLib).
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_paths, dim))


# ---------------------------------------------------------------------------
# 3. Path simulation: bridge per factor, correlate increments, build GBM paths
# ---------------------------------------------------------------------------
def simulate_worst_performance(n_paths, times, spots, vols, q, r, corr,
                               seed=42, batch=20000):
    """
    Returns (n_paths, m) array of worst-of performance min_i S_i(t_k)/S0_i at each date.
    Streams in batches to cap memory (N*m*d floats per batch).
    """
    times = np.asarray(times, float)
    spots = np.asarray(spots, float)
    vols = np.asarray(vols, float)
    q = np.asarray(q, float)
    m, d = len(times), len(spots)
    D = m * d
    dt = np.diff(np.concatenate(([0.0], times)))          # (m,)
    drift = (r - q - 0.5 * vols ** 2)                     # (d,)
    L = np.linalg.cholesky(corr)                          # lower, L L^T = corr

    bb = BrownianBridge(times)
    Z_all = sobol_normals(n_paths, D, seed=seed)          # (N, D)

    worst = np.empty((n_paths, m))
    for s in range(0, n_paths, batch):
        Z = Z_all[s:s + batch]
        nb = Z.shape[0]
        # reshape so bridge-step major, factor minor: dim index = step*d + f
        Z = Z.reshape(nb, m, d)
        # bridge each factor across time -> levels W (time order)
        W = np.empty((nb, m, d))
        for f in range(d):
            W[:, :, f] = bb.transform(Z[:, :, f])
        # independent increments dW (var dt_k per step), then correlate across factors
        dW = np.diff(W, axis=1, prepend=0.0)              # (nb, m, d)
        dW_corr = dW @ L.T                                # Cov per step = dt_k * corr
        # exact log-Euler GBM (constant params)
        log_incr = drift[None, None, :] * dt[None, :, None] + vols[None, None, :] * dW_corr
        logS = np.log(spots)[None, None, :] + np.cumsum(log_incr, axis=1)
        S = np.exp(logS)                                  # (nb, m, d)
        worst[s:s + nb] = np.min(S / spots[None, None, :], axis=2)
    return worst


# ---------------------------------------------------------------------------
# 4. A representative worst-of autocall payoff (adapt to your real termsheet)
# ---------------------------------------------------------------------------
def price_autocall(worst, times, r,
                   autocall_barrier=1.00,   # redeem early if worst >= this on an obs date
                   coupon=0.07,             # annualised coupon, paid on redemption
                   protection_barrier=0.60, # at maturity: capital protected if worst >= this
                   notional=1.0):
    times = np.asarray(times, float)
    N, m = worst.shape
    df = np.exp(-r * times)
    payoff_pv = np.zeros(N)
    alive = np.ones(N, dtype=bool)

    for k in range(m):
        if k < m - 1:
            trig = alive & (worst[:, k] >= autocall_barrier)
            payoff_pv[trig] = notional * (1.0 + coupon * times[k]) * df[k]
            alive &= ~trig
        else:  # maturity
            w = worst[alive, k]
            redeemed = w >= autocall_barrier
            protected = (~redeemed) & (w >= protection_barrier)
            lossy = (~redeemed) & (~protected)
            idx = np.where(alive)[0]
            payoff_pv[idx[redeemed]] = notional * (1.0 + coupon * times[k]) * df[k]
            payoff_pv[idx[protected]] = notional * df[k]
            payoff_pv[idx[lossy]] = notional * w[lossy] * df[k]   # capital at risk

    price = payoff_pv.mean()
    stderr = payoff_pv.std(ddof=1) / np.sqrt(N)
    return price, stderr


# ---------------------------------------------------------------------------
# 5. ACCEPTANCE TEST - run this first. Validates bridge math + time scaling.
#    Sample covariance of (W(t_j), W(t_k)) must approximate min(t_j, t_k).
# ---------------------------------------------------------------------------
def verify_bridge_covariance(times, n=200000, seed=1):
    times = np.asarray(times, float)
    m = len(times)
    bb = BrownianBridge(times)
    rng = np.random.default_rng(seed)            # iid normals: bridge must reproduce BM
    Z = rng.standard_normal((n, m))
    W = bb.transform(Z)
    emp = (W.T @ W) / n
    theo = np.minimum.outer(times, times)
    max_abs_err = np.max(np.abs(emp - theo))
    print("max |empirical cov - min(s,t)| =", max_abs_err)
    print("PASS" if max_abs_err < 0.02 else "FAIL (check bridge construction)")
    return max_abs_err


if __name__ == "__main__":
    # quarterly observations for 3 years
    times = np.array([0.25 * i for i in range(1, 13)])

    print("== bridge acceptance test ==")
    verify_bridge_covariance(times)

    print("\n== example 3-asset worst-of autocall ==")
    d = 3
    spots = [100.0] * d
    vols = [0.25, 0.30, 0.22]
    q = [0.0, 0.0, 0.0]
    r = 0.03
    corr = np.array([[1.0, 0.5, 0.4],
                     [0.5, 1.0, 0.45],
                     [0.4, 0.45, 1.0]])
    worst = simulate_worst_performance(50000, times, spots, vols, q, r, corr)
    price, se = price_autocall(worst, times, r)
    print(f"price = {price:.4f}  std.err = {se:.4f}")
    print("Compare Sobol+bridge vs Sobol+forward vs pseudo-random at matched paths")
    print("to see the convergence benefit (and check Greeks, which are more sensitive).")
