import numpy as np


class PowerCurve:
    """
    Wind turbine power curve: maps wind speed to power output.

    Two modes:
      - Parametric (default): cut-in / rated speed / cut-out / rated power,
        with a simplified cubic ramp-up (see `power()` below).
      - Table-based: an uploaded (wind_speed, power) curve, e.g. copied from
        a manufacturer's datasheet as a CSV. Values are linearly
        interpolated between table points; outside the table's speed range
        the power is 0.
    """

    def __init__(self,
                 cut_in: float = 3.0,
                 rated_speed: float = 11.0,
                 cut_out: float = 25.0,
                 rated_power: float = 2500.0,
                 speed_table=None,
                 power_table=None):
        """
        Parameters
        ----------
        cut_in, rated_speed, cut_out, rated_power :
            Parametric power curve inputs (ignored if a table is supplied).
        speed_table, power_table : array-like, optional
            If both are given, the power curve uses linear interpolation
            over this table instead of the parametric cubic model.
        """
        self.cut_in = cut_in
        self.rated_speed = rated_speed
        self.cut_out = cut_out

        if speed_table is not None and power_table is not None:
            speed_table = np.asarray(speed_table, dtype=float)
            power_table = np.asarray(power_table, dtype=float)
            order = np.argsort(speed_table)
            self.speed_table = speed_table[order]
            self.power_table = power_table[order]
            self.rated_power = float(np.max(self.power_table))
        else:
            self.speed_table = None
            self.power_table = None
            self.rated_power = rated_power

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_table(cls, df):
        """
        Build a table-based PowerCurve from an uploaded CSV with two
        columns: wind speed [m/s] and power [kW]. Column names are matched
        loosely and case-insensitively.
        """
        cols = {c.lower().strip(): c for c in df.columns}

        def find_col(*keywords):
            for lower_name, original in cols.items():
                if all(kw in lower_name for kw in keywords):
                    return original
            return None

        speed_col = find_col("speed") or find_col("vitesse") or find_col("wind")
        power_col = find_col("power") or find_col("puissance")

        if speed_col is None or power_col is None:
            raise ValueError(
                f"CSV de courbe de puissance : colonnes non reconnues. "
                f"Attendu une colonne vitesse (ex: 'wind_speed') et une colonne "
                f"puissance (ex: 'power_kw'). Colonnes trouvées : {list(df.columns)}"
            )

        speeds = df[speed_col].astype(float).values
        powers = df[power_col].astype(float).values
        return cls(speed_table=speeds, power_table=powers)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def power(self, wind_speed: float) -> float:
        """
        Compute power output for a given wind speed.

        Parameters
        ----------
        wind_speed : float
            Wind speed at hub height [m/s].

        Returns
        -------
        float
            Power output [kW].
        """
        if self.speed_table is not None:
            if wind_speed < self.speed_table[0] or wind_speed > self.speed_table[-1]:
                return 0.0
            return float(np.interp(wind_speed, self.speed_table, self.power_table))

        if wind_speed < self.cut_in or wind_speed > self.cut_out:
            return 0.0

        if wind_speed <= self.rated_speed:
            # Cubic relationship below rated speed (simplified)
            return self.rated_power * (
                (wind_speed - self.cut_in) /
                (self.rated_speed - self.cut_in)
            ) ** 3
        else:
            # Constant power above rated speed
            return self.rated_power

    def __repr__(self) -> str:
        if self.speed_table is not None:
            return (
                f"PowerCurve(table, {len(self.speed_table)} points, "
                f"P_rated={self.rated_power:.0f} kW)"
            )
        return (
            f"PowerCurve(cut_in={self.cut_in}, "
            f"rated={self.rated_speed}, cut_out={self.cut_out}, "
            f"P_rated={self.rated_power} kW)"
        )
