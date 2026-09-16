from dataclasses import dataclass
import math


@dataclass
class Turbine:
    id: str
    x: float
    y: float
    diameter: float
    hub_height: float

    def __post_init__(self):
        if self.diameter <= 0:
            raise ValueError(f"Turbine {self.id}: rotor diameter must be > 0")
        if self.hub_height <= 0:
            raise ValueError(f"Turbine {self.id}: hub height must be > 0")

        self.u_inflow: float = None
        self.u_effective: float = None
        self.power_kw: float = None

    @property
    def swept_area(self) -> float:
        return math.pi * (self.diameter / 2.0) ** 2

    def __repr__(self) -> str:
        return (
            f"Turbine(id={self.id!r}, x={self.x:.1f}, y={self.y:.1f}, "
            f"D={self.diameter:.1f} m, H={self.hub_height:.1f} m)"
        )


if __name__ == "__main__":
    t = Turbine(id="T01", x=0.0, y=0.0, diameter=120.0, hub_height=90.0)
    print(t)
    print("Swept area:", t.swept_area)

"""
wind_farm.py
============

Defines the `WindFarm` class: a collection of Turbine objects plus the
current ambient wind conditions (speed and direction), and the geometric
machinery needed to reason about "who is upstream of whom".

Engineering background
-----------------------
A wake model needs to answer, for every ordered pair of turbines (i, j):

    "If turbine i is upstream, how far downstream (xi) and how far off
     to the side (eta) does that put turbine j, in a frame aligned with
     the wind rather than with the map's North/East axes?"

This requires converting the farm's map coordinates (Easting, Northing)
into a "wind-aligned" frame. This module implements exactly that
transform, plus basic farm bookkeeping (add turbines, set wind
conditions).

Meteorological convention
--------------------------
Wind direction `theta` is given in the meteorological convention: the
compass bearing the wind is blowing FROM, in degrees clockwise from
North (0 deg = wind from the North, 90 deg = wind from the East, etc.).
This matches how wind roses and met-mast data are normally reported, and
how WAsP expects direction sectors to be defined.

Derivation of the rotation
---------------------------
The wind travels TOWARD compass bearing:

    theta_wind = theta + 180 deg

In a plane with x = East, y = North, the unit vector pointing toward a
compass bearing theta_wind (clockwise from North) is:

    d_hat = ( sin(theta_wind), cos(theta_wind) )

    Sanity check: theta_wind = 0   -> (0, 1)  = due North   (correct)
                  theta_wind = 90  -> (1, 0)  = due East    (correct)

The crosswind unit vector, perpendicular to d_hat (rotated +90 deg from
it) is:

    c_hat = ( cos(theta_wind), -sin(theta_wind) )

For any two turbines i (candidate upstream) and j (candidate downstream),
with map-frame separation vector:

    delta = (x_j - x_i, y_j - y_i)

the downstream and crosswind distances are simply the projections:

    xi_ij  = delta . d_hat      (positive => j is downwind of i)
    eta_ij = delta . c_hat      (0 => j is directly in i's wake centerline)

This is the geometric backbone the Eddy Viscosity wake model (Phase 2)
will use: it needs xi_ij to evaluate wake expansion/deficit as a function
of downstream distance, and eta_ij to evaluate how far off the wake
centerline turbine j sits.

Assumptions
-----------
- Flat terrain (no orographic speed-up), single wind direction / speed
  applied uniformly across the whole farm at any one time (real sites see
  the wind direction/speed vary slightly across a large farm -- we ignore
  that here, as does a first-pass WAsP run before terrain correction).
- No wind veer with height (direction assumed constant from surface to
  hub height).
"""

import numpy as np
from turbine import Turbine


class WindFarm:
    """
    A collection of turbines plus the ambient wind condition currently
    being simulated.

    Parameters
    ----------
    name : str
        Name of the wind farm project.
    """

    def __init__(self, name: str = "Untitled Wind Farm"):
        self.name = name
        self.turbines: list[Turbine] = []

        # Ambient (free-stream, undisturbed) conditions -- set via
        # set_wind_condition(). None until the user sets them explicitly,
        # so mistakes (running a simulation before specifying wind) fail
        # loudly rather than silently defaulting to something.
        self.wind_speed: float = None       # U_inf [m/s]
        self.wind_direction: float = None   # theta [deg], meteorological

    # ------------------------------------------------------------------
    # Farm construction
    # ------------------------------------------------------------------
    def add_turbine(self, turbine: Turbine) -> None:
        """Add a Turbine to the farm, checking for duplicate IDs."""
        if any(t.id == turbine.id for t in self.turbines):
            raise ValueError(f"Turbine id '{turbine.id}' already exists in farm.")
        self.turbines.append(turbine)

    def add_turbines(self, turbines: list[Turbine]) -> None:
        for t in turbines:
            self.add_turbine(t)

    def set_wind_condition(self, wind_speed: float, wind_direction: float) -> None:
        """
        Set the free-stream wind speed and direction for the next
        simulation.

        Parameters
        ----------
        wind_speed : float
            Free-stream wind speed U_inf [m/s]. Must be > 0.
        wind_direction : float
            Meteorological wind direction theta [deg], the compass
            bearing the wind is blowing FROM (0-360, clockwise from
            North).
        """
        if wind_speed <= 0:
            raise ValueError("wind_speed must be > 0")
        wind_direction = wind_direction % 360.0
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

    # ------------------------------------------------------------------
    # Geometry: map frame -> wind-aligned frame
    # ------------------------------------------------------------------
    def wind_unit_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (d_hat, c_hat): the downstream and crosswind unit vectors
        in the (East, North) map frame, for the farm's currently set
        wind_direction.

        Returns
        -------
        d_hat : np.ndarray, shape (2,)
            Unit vector pointing in the direction the wind travels TOWARD.
        c_hat : np.ndarray, shape (2,)
            Unit vector perpendicular to d_hat (crosswind direction).
        """
        if self.wind_direction is None:
            raise RuntimeError("wind_direction not set. Call set_wind_condition() first.")

        theta_wind = np.deg2rad(self.wind_direction + 180.0)
        d_hat = np.array([np.sin(theta_wind), np.cos(theta_wind)])
        c_hat = np.array([np.cos(theta_wind), -np.sin(theta_wind)])
        return d_hat, c_hat

    def wind_aligned_distances(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the full pairwise downstream/crosswind distance matrices
        for every ordered pair of turbines (i, j) in the farm.

        Returns
        -------
        xi : np.ndarray, shape (N, N)
            xi[i, j] = downstream distance from turbine i to turbine j
            [m]. Positive means j is downwind of i (i.e. i could
            potentially wake j).
        eta : np.ndarray, shape (N, N)
            eta[i, j] = crosswind offset from turbine i's wake centerline
            to turbine j [m]. 0 means j sits exactly on i's centerline.

        Notes
        -----
        This is O(N^2) in the number of turbines, which is fine for the
        farm sizes (tens to low hundreds of turbines) this project
        targets. Real commercial tools use the same approach.
        """
        d_hat, c_hat = self.wind_unit_vectors()

        positions = np.array([[t.x, t.y] for t in self.turbines])  # (N, 2)
        n = len(self.turbines)

        # delta[i, j] = position[j] - position[i], vector shape (N, N, 2)
        delta = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

        xi = delta @ d_hat    # (N, N)
        eta = delta @ c_hat   # (N, N)

        return xi, eta

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"WindFarm(name={self.name!r}, n_turbines={len(self.turbines)}, "
            f"U_inf={self.wind_speed}, theta={self.wind_direction})"
        )
    """
visualization.py
================

All plotting functions live here, kept separate from the physics/geometry
modules so that (a) the physics code has no plotting dependency and can
be unit-tested headlessly, and (b) the GUI (Phase 5) can reuse the same
plotting functions on an embedded Matplotlib canvas instead of duplicating
drawing logic.

This file currently implements only the Phase 1 requirement: a 2D map of
turbine positions, with a wind direction indicator and a North arrow
(standard convention in wind resource assessment plots, matching how
WAsP displays its site maps).
"""

import numpy as np
import matplotlib.pyplot as plt
from wind_farm import WindFarm


def plot_layout(farm: WindFarm, ax=None, show_labels: bool = True):
    """
    Plot the 2D layout of a WindFarm: turbine positions (with rotor
    diameter drawn to scale), a North arrow, and -- if wind conditions
    have been set -- an arrow showing the wind direction.

    Parameters
    ----------
    farm : WindFarm
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure/axes is created.
    show_labels : bool
        Whether to label each turbine with its id.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    xs = np.array([t.x for t in farm.turbines])
    ys = np.array([t.y for t in farm.turbines])
    diam = np.array([t.diameter for t in farm.turbines])

    # Draw each turbine as a circle scaled to its actual rotor radius,
    # so relative spacing (in rotor diameters -- the physically relevant
    # unit for wake interaction) is visually apparent.
    for t in farm.turbines:
        circle = plt.Circle((t.x, t.y), radius=t.diameter / 2.0,
                             facecolor="steelblue", edgecolor="black",
                             linewidth=1.0, alpha=0.8, zorder=3)
        ax.add_patch(circle)
        if show_labels:
            ax.annotate(t.id, (t.x, t.y), textcoords="offset points",
                        xytext=(0, diam.max() * 0.6 if len(diam) else 10),
                        ha="center", fontsize=9, zorder=4)

    ax.scatter(xs, ys, s=10, color="black", zorder=5)

    # --- Axis framing ---
    pad = max(diam.max() * 4, 50) if len(diam) else 50
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title(f"{farm.name} -- Layout")
    ax.grid(True, linestyle=":", alpha=0.5)

    # --- North arrow (always shown, independent of wind) ---
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    nx = x1 - 0.08 * (x1 - x0)
    ny = y1 - 0.20 * (y1 - y0)
    arrow_len = 0.08 * (y1 - y0)
    ax.annotate("N", xy=(nx, ny + arrow_len), xytext=(nx, ny),
                arrowprops=dict(facecolor="black", width=2, headwidth=8),
                ha="center", fontsize=11, fontweight="bold")

    # --- Wind direction arrow (only if wind condition is set) ---
    if farm.wind_direction is not None:
        d_hat, _ = farm.wind_unit_vectors()
        wx = x0 + 0.12 * (x1 - x0)
        wy = y1 - 0.20 * (y1 - y0)
        arrow_len_w = 0.10 * (y1 - y0)
        ax.annotate(
            "", xy=(wx + d_hat[0] * arrow_len_w, wy + d_hat[1] * arrow_len_w),
            xytext=(wx, wy),
            arrowprops=dict(facecolor="firebrick", edgecolor="firebrick", width=2, headwidth=9),
        )
        ax.text(wx, wy - arrow_len_w * 0.4,
                 f"Wind: {farm.wind_direction:.0f}°  ({farm.wind_speed:.1f} m/s)",
                 color="firebrick", fontsize=9, ha="left")

    return ax
"""
demo_phase1.py
==============

Small end-to-end demo of Phase 1: build a farm, set wind conditions,
inspect the wind-aligned distance matrices, and plot the layout.

Layout used: a simple 3x2 grid, turbines spaced 5D apart in both
directions (5D is a fairly typical row spacing in real wind farms,
chosen so wake interactions -- which we'll model next phase -- will be
visually and physically meaningful).
"""

import numpy as np
import matplotlib.pyplot as plt

from turbine import Turbine
from wind_farm import WindFarm
from visualization import plot_layout

D = 90.0   # rotor diameter [m] -- roughly a 2-3 MW class turbine
H = 80.0   # hub height [m]
SPACING = 5 * D  # 5 rotor diameters between turbines

farm = WindFarm(name="Demo Farm - 3x2 grid")

turbines = []
tid = 1
for row in range(2):        # 2 rows (North-South)
    for col in range(3):    # 3 columns (East-West)
        x = col * SPACING
        y = row * SPACING
        turbines.append(Turbine(id=f"T{tid:02d}", x=x, y=y,
                                 diameter=D, hub_height=H))
        tid += 1

farm.add_turbines(turbines)

# Wind blowing FROM the West (270 deg meteorological) -> travels East.
# With this layout, this should put T02/T03 directly downstream of
# T01/T02 respectively (same row), and produce zero East-West stagger.
farm.set_wind_condition(wind_speed=9.0, wind_direction=270.0)

print(farm)
print()
for t in farm.turbines:
    print(t)
print()

xi, eta = farm.wind_aligned_distances()
np.set_printoptions(precision=1, suppress=True)
print("Downstream distance matrix xi [m] (rows=upstream candidate i, cols=j):")
print(xi)
print()
print("Crosswind distance matrix eta [m]:")
print(eta)
print()

# Quick sanity check: for wind from the West, T01->T02 should be purely
# downstream (xi = SPACING) with zero crosswind offset (eta = 0), since
# they're in the same row (same y).
i, j = 0, 1  # T01 -> T02
print(f"Sanity check T01->T02: xi={xi[i,j]:.1f} m (expect {SPACING:.1f} m), "
      f"eta={eta[i,j]:.1f} m (expect 0.0 m)")

ax = plot_layout(farm)
plt.tight_layout()
plt.savefig("phase1_layout.png", dpi=150)
print("\nSaved plot to phase1_layout.png")
