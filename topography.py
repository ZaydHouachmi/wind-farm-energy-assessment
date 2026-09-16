"""
topography.py
==============
Grid-based terrain elevation surface for the wind farm site.

This is a *reporting* elevation model: it lets you attach a ground level
Z to each turbine (x, y) so the results table can show real hub
elevations (x, y, z). It does not drive a terrain speed-up / CFD model;
the wake models in wake_model.py remain flat-terrain models. Extending
to terrain-induced flow effects would be a separate, larger project.

Three ways to get a Topography object:
  - Topography.flat(...)            constant elevation
  - Topography.synthetic_hill(...)  a single Gaussian hill/ridge, for demos
  - Topography.from_points(x, y, z) interpolate real/measured terrain points
"""
from __future__ import annotations
import numpy as np


class Topography:
    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray):
        """
        x_grid, y_grid : 1D arrays defining the grid axes [m]
        z_grid : 2D array, shape (len(y_grid), len(x_grid)), elevation [m]
        """
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.y_grid = np.asarray(y_grid, dtype=float)
        self.z_grid = np.asarray(z_grid, dtype=float)
        self._interp = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def flat(cls, x_min, x_max, y_min, y_max, elevation: float = 0.0, n: int = 50):
        xs = np.linspace(x_min, x_max, n)
        ys = np.linspace(y_min, y_max, n)
        zg = np.full((n, n), float(elevation))
        return cls(xs, ys, zg)

    @classmethod
    def synthetic_hill(cls, x_min, x_max, y_min, y_max,
                        base_elevation: float = 0.0, hill_height: float = 60.0,
                        hill_cx: float | None = None, hill_cy: float | None = None,
                        hill_sigma: float | None = None, n: int = 80):
        xs = np.linspace(x_min, x_max, n)
        ys = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(xs, ys)
        cx = hill_cx if hill_cx is not None else (x_min + x_max) / 2.0
        cy = hill_cy if hill_cy is not None else (y_min + y_max) / 2.0
        sigma = hill_sigma if hill_sigma is not None else max(x_max - x_min, y_max - y_min) / 4.0
        sigma = max(sigma, 1e-6)
        Z = base_elevation + hill_height * np.exp(
            -(((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))
        )
        return cls(xs, ys, Z)

    @classmethod
    def from_points(cls, x, y, z, n: int = 80):
        """Interpolate scattered survey points (x, y, z) onto a regular grid."""
        from scipy.interpolate import griddata
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        xs = np.linspace(x.min(), x.max(), n)
        ys = np.linspace(y.min(), y.max(), n)
        X, Y = np.meshgrid(xs, ys)
        Z = griddata((x, y), z, (X, Y), method="linear")
        Z_nearest = griddata((x, y), z, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Z_nearest, Z)
        return cls(xs, ys, Z)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def _get_interpolator(self):
        if self._interp is None:
            from scipy.interpolate import RegularGridInterpolator
            self._interp = RegularGridInterpolator(
                (self.y_grid, self.x_grid), self.z_grid,
                bounds_error=False, fill_value=None,
            )
        return self._interp

    def elevation_at(self, x, y) -> float:
        """Elevation at a single point (x, y), extrapolated at the edges."""
        interp = self._get_interpolator()
        val = interp(np.array([[float(y), float(x)]]))
        return float(val[0])

    def elevation_grid(self):
        """Return (X, Y, Z) meshgrid arrays for contour/heatmap plotting."""
        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        return X, Y, self.z_grid
