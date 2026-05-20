"""Sobol low-discrepancy + Brownian-bridge multi-asset path generator.

Pipeline
--------
1. Build a Sobol uniform sequence of dimension ``N_assets * n_steps`` (default
   Joe-Kuo D7 direction integers).
2. Wrap in an inverse-CDF Gaussian sequence generator
   (``GaussianLowDiscrepancySequenceGenerator``).
3. For each path: reshape the ``N*n_steps`` standard normals into a
   ``(N_assets, n_steps)`` matrix; one row per asset.
4. **Brownian-bridge construction per asset** via ``ql.BrownianBridge`` so the
   most important Sobol coordinates govern the largest-variance time scales.
   ``ql.BrownianBridge.transform`` returns standard-normal variates per step
   (it reshuffles the Sobol coordinates through the bridge weights; the
   marginal of each output entry is N(0, 1)), which is exactly what
   ``StochasticProcess.evolve`` expects.
5. Step the ``StochasticProcessArray`` manually: at each time-step pass an
   ``ql.Array`` of N uncorrelated standard normals; the array applies the
   Cholesky factor of the asset-correlation matrix internally.

Why not ``GaussianSobolMultiPathGenerator(... brownianBridge=True)``?
    QuantLib's general SDE multipath generator does not implement Brownian-
    bridge time construction for that template instantiation (it raises
    "Brownian bridge not supported" at sample time). The market-model code
    path supports it but is not appropriate for a Local-Vol multi-asset SDE.

Notes
-----
* Antithetic sampling with Sobol is generally not used because Sobol is a
  deterministic balanced sampler; the flag is exposed but defaults to ``False``.
* The same ``seed`` produces identical output across runs.
"""

from __future__ import annotations

import numpy as np
import QuantLib as ql


_DIRECTION_INTEGERS = {
    "Jaeckel": ql.SobolRsg.Jaeckel,
    "JoeKuoD7": ql.SobolRsg.JoeKuoD7,
    "SobolLevitan": ql.SobolRsg.SobolLevitan,
}


class PathGenerator:
    """Generate correlated multi-asset paths under Local-Vol with Sobol + BB."""

    def __init__(
        self,
        process_array: ql.StochasticProcessArray,
        time_grid: ql.TimeGrid,
        n_paths: int,
        seed: int = 42,
        bridge: bool = True,
        direction_integers: str = "JoeKuoD7",
        antithetic: bool = False,
    ) -> None:
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if direction_integers not in _DIRECTION_INTEGERS:
            raise ValueError(
                f"direction_integers must be one of {list(_DIRECTION_INTEGERS)}"
            )

        self.process_array = process_array
        self.time_grid = time_grid
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self.bridge = bool(bridge)
        self.antithetic = bool(antithetic)

        self.n_assets = process_array.size()
        self.n_steps = len(time_grid) - 1
        self.dimension = self.n_assets * self.n_steps

        self._times = np.array(
            [time_grid[i] for i in range(len(time_grid))], dtype=float
        )
        self._dts = np.diff(self._times)

        uniform = ql.UniformLowDiscrepancySequenceGenerator(
            self.dimension, self.seed, _DIRECTION_INTEGERS[direction_integers]
        )
        self._gsg = ql.GaussianLowDiscrepancySequenceGenerator(uniform)
        # The first Sobol point is the origin, which inverse-CDF maps to 0
        # for every coordinate. Drop it so paths are non-degenerate.
        self._gsg.nextSequence()
        self._bb = ql.BrownianBridge(time_grid) if self.bridge else None

        self._x0 = np.array(list(process_array.initialValues()), dtype=float)

    def _draw_normals(self) -> np.ndarray:
        """Draw the next Sobol-Gaussian sample, shape (N_assets, n_steps)."""
        z = np.asarray(self._gsg.nextSequence().value(), dtype=float)
        return z.reshape(self.n_assets, self.n_steps)

    def _bridge_to_normals(self, z: np.ndarray) -> np.ndarray:
        """Per asset: BB-transform Sobol normals → reshuffled N(0,1) per step."""
        out = np.empty_like(z)
        for a in range(self.n_assets):
            out[a] = np.asarray(self._bb.transform(z[a].tolist()), dtype=float)
        return out

    def _step_one_path(self, dw_normals: np.ndarray) -> np.ndarray:
        """Evolve the process along one path using pre-drawn standard normals.

        ``dw_normals`` has shape (N_assets, n_steps). The process array applies
        Cholesky correlation internally on each step.
        """
        n_nodes = self.n_steps + 1
        path = np.empty((self.n_assets, n_nodes), dtype=float)
        path[:, 0] = self._x0

        x = ql.Array(self.n_assets)
        for a in range(self.n_assets):
            x[a] = float(self._x0[a])

        dw = ql.Array(self.n_assets)
        for k in range(self.n_steps):
            for a in range(self.n_assets):
                dw[a] = float(dw_normals[a, k])
            x = self.process_array.evolve(
                float(self._times[k]), x, float(self._dts[k]), dw
            )
            for a in range(self.n_assets):
                path[a, k + 1] = x[a]
        return path

    def generate(self) -> np.ndarray:
        """Return all paths as (n_paths, n_assets, n_steps + 1)."""
        n_nodes = self.n_steps + 1
        out = np.empty((self.n_paths, self.n_assets, n_nodes), dtype=float)
        for p in range(self.n_paths):
            z = self._draw_normals()
            if self.antithetic and p % 2 == 1:
                z = -z
            dw_n01 = self._bridge_to_normals(z) if self.bridge else z
            out[p] = self._step_one_path(dw_n01)
        return out

    def generate_iter(self):
        """Yield (path_index, np.ndarray of shape (n_assets, n_steps+1)) lazily."""
        for p in range(self.n_paths):
            z = self._draw_normals()
            if self.antithetic and p % 2 == 1:
                z = -z
            dw_n01 = self._bridge_to_normals(z) if self.bridge else z
            yield p, self._step_one_path(dw_n01)
