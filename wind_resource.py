"""
wind_resource.py
=================
Directional wind resource: a wind rose (frequency by sector) plus a
Weibull wind-speed distribution per sector. Used to integrate the power
curve (and wake losses) over the full joint speed/direction distribution
to get an annual energy estimate, instead of a single design point.

Three ways to build a WindRose:
  - WindRose.default_12_sector(...)     synthetic bell-shaped rose, for demos
  - WindRose.from_table(df)             a pre-computed rose table (direction,
                                          frequency, Weibull k, Weibull A),
                                          e.g. uploaded as a CSV
  - WindRose.from_measurements(dir, sp) raw time-series measurements
                                          (direction, speed pairs), binned
                                          into sectors and Weibull-fitted
"""
from __future__ import annotations
import numpy as np


class WindRose:
    def __init__(self, sector_centers_deg, frequencies, weibull_k, weibull_a):
        """
        sector_centers_deg : directional sector centers [deg], e.g. 12 sectors -> 0,30,...,330
        frequencies        : fraction of time wind blows FROM each sector (normalized to sum=1)
        weibull_k          : Weibull shape parameter per sector [-]
        weibull_a          : Weibull scale parameter per sector [m/s]
        """
        self.sector_centers_deg = np.asarray(sector_centers_deg, dtype=float)
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.weibull_k = np.asarray(weibull_k, dtype=float)
        self.weibull_a = np.asarray(weibull_a, dtype=float)

        total = self.frequencies.sum()
        if total <= 0:
            raise ValueError("Sector frequencies must sum to a positive value")
        if abs(total - 1.0) > 1e-6:
            self.frequencies = self.frequencies / total  # normalize defensively

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def default_12_sector(cls, prevailing_deg: float = 270.0, prevailing_speed: float = 9.0,
                          spread_deg: float = 60.0, k: float = 2.0):
        """A simple bell-shaped rose around one prevailing direction, for demo purposes."""
        centers = np.arange(0.0, 360.0, 30.0)
        diff = (centers - prevailing_deg + 180.0) % 360.0 - 180.0
        weights = np.exp(-0.5 * (diff / spread_deg) ** 2) + 0.05
        freqs = weights / weights.sum()
        a = np.full_like(centers, prevailing_speed)
        kk = np.full_like(centers, k)
        return cls(centers, freqs, kk, a)

    @classmethod
    def from_table(cls, df):
        """
        Build a WindRose from a pre-computed rose table, e.g. an uploaded CSV
        with columns for direction, frequency, Weibull k and Weibull A.
        Column names are matched case-insensitively and loosely (so
        "Direction [deg]", "direction_deg", "dir" etc. all work).
        """
        cols = {c.lower().strip(): c for c in df.columns}

        def find_col(*keywords):
            for lower_name, original in cols.items():
                if all(kw in lower_name for kw in keywords):
                    return original
            return None

        dir_col = find_col("dir")
        freq_col = find_col("freq")
        k_col = find_col("k")
        a_col = find_col("a")

        missing = [name for name, col in
                   [("direction", dir_col), ("frequency", freq_col),
                    ("weibull k", k_col), ("weibull A", a_col)] if col is None]
        if missing:
            raise ValueError(
                f"CSV de rose des vents : colonnes manquantes ou non reconnues pour {missing}. "
                f"Colonnes trouvées : {list(df.columns)}"
            )

        directions = df[dir_col].astype(float).values
        freq_raw = df[freq_col].astype(float).values
        # Accept either fractions (sum ~1) or percentages (sum ~100)
        if freq_raw.sum() > 1.5:
            freq_raw = freq_raw / 100.0
        k_vals = df[k_col].astype(float).values
        a_vals = df[a_col].astype(float).values

        return cls(directions, freq_raw, k_vals, a_vals)

    @classmethod
    def from_measurements(cls, directions, speeds, n_sectors: int = 12,
                           min_samples_per_sector: int = 3):
        """
        Build a WindRose from raw (direction, speed) time-series measurements,
        e.g. an uploaded met-mast CSV.

        Directions are binned into `n_sectors` equal-width sectors centered
        on 0, 360/n, 2*360/n, ... The Weibull k and A per sector are
        estimated from the sample mean and standard deviation of speeds in
        that sector using the Justus moment-based approximation:

            k ~= (sigma / mu) ** -1.086
            A = mu / Gamma(1 + 1/k)

        Sectors with fewer than `min_samples_per_sector` observations fall
        back to the overall-sample mean speed and k=2.0, with their
        frequency computed normally (so a nearly-calm sector still gets a
        small, correctly weighted probability mass).
        """
        from scipy.special import gamma

        directions = np.asarray(directions, dtype=float) % 360.0
        speeds = np.asarray(speeds, dtype=float)
        valid = np.isfinite(directions) & np.isfinite(speeds) & (speeds >= 0)
        directions = directions[valid]
        speeds = speeds[valid]
        n_total = len(speeds)
        if n_total == 0:
            raise ValueError("Aucune mesure valide (direction, vitesse) trouvée dans le fichier.")

        sector_width = 360.0 / n_sectors
        centers = np.arange(0.0, 360.0, sector_width)

        # Assign each measurement to its nearest sector center
        sector_idx = np.round(directions / sector_width).astype(int) % n_sectors

        overall_mean = float(np.mean(speeds))
        frequencies = np.zeros(n_sectors)
        weibull_k = np.full(n_sectors, 2.0)
        weibull_a = np.full(n_sectors, max(overall_mean, 0.1))

        for s in range(n_sectors):
            sector_speeds = speeds[sector_idx == s]
            frequencies[s] = len(sector_speeds) / n_total
            if len(sector_speeds) >= min_samples_per_sector:
                mu = float(np.mean(sector_speeds))
                sigma = float(np.std(sector_speeds))
                if mu > 1e-6 and sigma > 1e-6:
                    k_est = (sigma / mu) ** (-1.086)
                    k_est = float(np.clip(k_est, 1.0, 3.5))  # keep in a physically sane range
                    a_est = mu / gamma(1.0 + 1.0 / k_est)
                    weibull_k[s] = k_est
                    weibull_a[s] = float(a_est)
                elif mu > 1e-6:
                    weibull_a[s] = mu

        return cls(centers, frequencies, weibull_k, weibull_a)

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------
    def weibull_pdf(self, speed, sector_idx: int):
        k = self.weibull_k[sector_idx]
        a = self.weibull_a[sector_idx]
        speed = np.asarray(speed, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pdf = (k / a) * (speed / a) ** (k - 1.0) * np.exp(-((speed / a) ** k))
        return np.where(speed > 0, pdf, 0.0)

    def mean_speed(self, sector_idx: int) -> float:
        from scipy.special import gamma
        k = self.weibull_k[sector_idx]
        a = self.weibull_a[sector_idx]
        return float(a * gamma(1.0 + 1.0 / k))

    def overall_mean_speed(self) -> float:
        return float(sum(self.frequencies[i] * self.mean_speed(i) for i in range(self.n_sectors())))

    def n_sectors(self) -> int:
        return len(self.sector_centers_deg)

    # ------------------------------------------------------------------
    # Roughness / wind profile (log law) height rescaling
    # ------------------------------------------------------------------
    def rescaled_to_height(self, target_height: float, roughness_z0: float, reference_height: float = 10.0):
        """
        Return a new WindRose with the Weibull scale parameter A rescaled from
        `reference_height` (typically the met-mast measurement height, e.g. 10 m)
        up to `target_height` (typically the turbine hub height), using the
        standard logarithmic wind profile:

            U(z) / U(zref) = ln(z / z0) / ln(zref / z0)

        The shape parameter k is left unchanged (a common simplifying
        assumption). This is a simplified stand-in for the terrain/roughness
        transposition that WAsP performs.
        """
        import math
        if roughness_z0 <= 0:
            raise ValueError("roughness_z0 must be > 0")
        factor = math.log(target_height / roughness_z0) / math.log(reference_height / roughness_z0)
        factor = max(factor, 0.01)
        new_a = self.weibull_a * factor
        return WindRose(self.sector_centers_deg, self.frequencies, self.weibull_k, new_a)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_wind_rose(self, ax=None):
        """Polar bar chart: bar direction = sector, radius = frequency [%],
        colour = mean speed in that sector."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="polar")

        theta = np.deg2rad(self.sector_centers_deg)
        width = 2 * np.pi / self.n_sectors() * 0.9
        radii = self.frequencies * 100.0
        speeds = np.array([self.mean_speed(i) for i in range(self.n_sectors())])

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise, meteorological convention
        bars = ax.bar(theta, radii, width=width, bottom=0.0)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=speeds.min(), vmax=max(speeds.max(), speeds.min() + 1e-6))
        for bar, speed in zip(bars, speeds):
            bar.set_facecolor(cmap(norm(speed)))
            bar.set_edgecolor("black")
            bar.set_alpha(0.9)
        ax.set_title("Wind rose (frequency % by sector,\ncolour = mean speed)", fontsize=10)
        return ax
