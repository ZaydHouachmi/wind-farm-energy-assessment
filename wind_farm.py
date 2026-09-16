import numpy as np
from turbine import Turbine


class WindFarm:
    """A collection of turbines plus ambient wind conditions."""

    def __init__(self, name: str = "Untitled Wind Farm"):
        self.name = name
        self.turbines: list[Turbine] = []
        self.wind_speed: float = None
        self.wind_direction: float = None

    def add_turbine(self, turbine: Turbine) -> None:
        if any(t.id == turbine.id for t in self.turbines):
            raise ValueError(f"Turbine id '{turbine.id}' already exists in farm.")
        self.turbines.append(turbine)

    def add_turbines(self, turbines: list[Turbine]) -> None:
        for t in turbines:
            self.add_turbine(t)

    def set_wind_condition(self, wind_speed: float, wind_direction: float) -> None:
        if wind_speed <= 0:
            raise ValueError("wind_speed must be > 0")
        wind_direction = wind_direction % 360.0
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

    def wind_unit_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self.wind_direction is None:
            raise RuntimeError("wind_direction not set. Call set_wind_condition() first.")

        theta_wind = np.deg2rad(self.wind_direction + 180.0)
        d_hat = np.array([np.sin(theta_wind), np.cos(theta_wind)])
        c_hat = np.array([np.cos(theta_wind), -np.sin(theta_wind)])
        return d_hat, c_hat

    def wind_aligned_distances(self) -> tuple[np.ndarray, np.ndarray]:
        d_hat, c_hat = self.wind_unit_vectors()
        positions = np.array([[t.x, t.y] for t in self.turbines])
        delta = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        xi = delta @ d_hat
        eta = delta @ c_hat
        return xi, eta

    def __repr__(self) -> str:
        return (
            f"WindFarm(name={self.name!r}, n_turbines={len(self.turbines)}, "
            f"U_inf={self.wind_speed}, theta={self.wind_direction})"
        )
