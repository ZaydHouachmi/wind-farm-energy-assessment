"""
demo_custom.py
==============
Customizable demo with different turbine layouts and configurations.
"""

import numpy as np
import matplotlib.pyplot as plt

from turbine import Turbine
from wind_farm import WindFarm
from visualization import plot_layout

# === Scenario 1: Single row (simple) ===
def demo_single_row():
    farm = WindFarm(name="Single Row - 5 turbines")
    
    D = 100.0
    H = 85.0
    spacing = 3 * D  # 3D spacing
    
    for i in range(5):
        turbines = [
            Turbine(id=f"T{i+1:02d}", x=i * spacing, y=0.0,
                   diameter=D, hub_height=H)
            for i in range(5)
        ]
    
    farm.add_turbines(turbines)
    farm.set_wind_condition(wind_speed=10.0, wind_direction=270.0)
    
    return farm

# === Scenario 2: L-shaped layout ===
def demo_l_shaped():
    farm = WindFarm(name="L-Shaped Layout")
    
    D = 95.0
    H = 80.0
    spacing = 4 * D
    
    turbines = []
    tid = 1
    
    # Horizontal row
    for i in range(4):
        turbines.append(Turbine(id=f"T{tid:02d}", x=i * spacing, y=0.0,
                               diameter=D, hub_height=H))
        tid += 1
    
    # Vertical column
    for j in range(1, 4):
        turbines.append(Turbine(id=f"T{tid:02d}", x=0.0, y=j * spacing,
                               diameter=D, hub_height=H))
        tid += 1
    
    farm.add_turbines(turbines)
    farm.set_wind_condition(wind_speed=8.5, wind_direction=45.0)
    
    return farm

# === Scenario 3: Mixed sizes ===
def demo_mixed_sizes():
    farm = WindFarm(name="Mixed Turbine Sizes")
    
    spacing = 400.0
    
    turbines = [
        Turbine(id="Large_1", x=0.0, y=0.0, diameter=150.0, hub_height=100.0),
        Turbine(id="Med_1", x=spacing, y=0.0, diameter=110.0, hub_height=85.0),
        Turbine(id="Small_1", x=2*spacing, y=0.0, diameter=80.0, hub_height=70.0),
        Turbine(id="Large_2", x=0.0, y=spacing, diameter=150.0, hub_height=100.0),
    ]
    
    farm.add_turbines(turbines)
    farm.set_wind_condition(wind_speed=9.5, wind_direction=180.0)
    
    return farm

# === Run demos ===
if __name__ == "__main__":
    scenarios = [
        ("single_row", demo_single_row()),
        ("l_shaped", demo_l_shaped()),
        ("mixed_sizes", demo_mixed_sizes()),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for (name, farm), ax in zip(scenarios, axes):
        print(f"\n{farm.name}")
        print(f"Turbines: {len(farm.turbines)}")
        for t in farm.turbines:
            print(f"  {t}")
        
        plot_layout(farm, ax=ax, show_labels=True)
    
    plt.tight_layout()
    plt.savefig("demo_custom_layouts.png", dpi=150)
    print("\nSaved to demo_custom_layouts.png")
