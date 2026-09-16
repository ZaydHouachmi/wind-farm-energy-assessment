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
