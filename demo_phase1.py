import numpy as np
import matplotlib.pyplot as plt

from turbine import Turbine
from wind_farm import WindFarm
from visualization import plot_layout

D = 90.0
H = 80.0
SPACING = 5 * D

farm = WindFarm(name="Demo Farm - 3x2 grid")

turbines = []
tid = 1
for row in range(2):
    for col in range(3):
        x = col * SPACING
        y = row * SPACING
        turbines.append(Turbine(id=f"T{tid:02d}", x=x, y=y,
                                 diameter=D, hub_height=H))
        tid += 1

farm.add_turbines(turbines)

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

i, j = 0, 1
print(f"Sanity check T01->T02: xi={xi[i,j]:.1f} m (expect {SPACING:.1f} m), "
      f"eta={eta[i,j]:.1f} m (expect 0.0 m)")

ax = plot_layout(farm)
plt.tight_layout()
plt.savefig("phase1_layout.png", dpi=150)
print("\nSaved plot to phase1_layout.png")
