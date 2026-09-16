"""
demo_with_wake.py
=================
Demo including wake deficits and power output calculation.
"""

import numpy as np
import matplotlib.pyplot as plt

from turbine import Turbine
from wind_farm import WindFarm
from visualization import plot_layout
from wake_model import SimpleWakeModel
from power_curve import PowerCurve

# Create farm
D = 90.0
H = 80.0
SPACING = 5 * D

farm = WindFarm(name="Demo with Wakes - 3x2 grid")

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

# Initialize wake model and power curve
wake_model = SimpleWakeModel(alpha=0.05, beta=0.5)
power_curve = PowerCurve(cut_in=3.0, rated_speed=11.0, cut_out=25.0,
                        rated_power=2500.0)

print(f"Farm: {farm}")
print(f"Wake model: {wake_model}")
print(f"Power curve: {power_curve}\n")

# Compute wind-aligned distances
xi, eta = farm.wind_aligned_distances()

# Compute effective speeds accounting for wakes
print("Turbine | Inflow [m/s] | Effective [m/s] | Deficit | Power [kW]")
print("-" * 65)

total_power = 0.0
for j, turbine in enumerate(farm.turbines):
    # Compute wake deficits from all upstream turbines
    deficits = []
    for i, upstream_turbine in enumerate(farm.turbines):
        if xi[i, j] > 0:  # i is upstream of j
            deficit = wake_model.wake_deficit(xi[i, j], eta[i, j], 
                                             upstream_turbine.diameter)
            deficits.append(deficit)
    
    # Effective speed
    if deficits:
        u_eff = wake_model.effective_speed(farm.wind_speed, np.array(deficits))
        deficit_total = 1.0 - (u_eff / farm.wind_speed)
    else:
        u_eff = farm.wind_speed
        deficit_total = 0.0
    
    # Power output
    power = power_curve.power(u_eff)
    total_power += power
    
    print(f"{turbine.id:8s} | {farm.wind_speed:12.2f} | {u_eff:15.2f} | "
          f"{deficit_total:7.2%} | {power:9.1f}")

print("-" * 65)
print(f"Total farm power: {total_power:.1f} kW")
print(f"Farm efficiency: {total_power / (len(farm.turbines) * power_curve.rated_power):.2%}")

# Plot
ax = plot_layout(farm)
plt.tight_layout()
plt.savefig("demo_with_wake_layout.png", dpi=150)
print("\nSaved plot to demo_with_wake_layout.png")
