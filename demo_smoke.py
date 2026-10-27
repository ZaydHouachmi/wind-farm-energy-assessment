"""Smoke test: compute total farm power for default configurations without GUI."""
import numpy as np
from turbine import Turbine
from wind_farm import WindFarm
from wake_model import SimpleWakeModel, JensenWakeModel, EddyViscosityWakeModel
from power_curve import PowerCurve


def make_farm(rows=2, cols=3, spacing=450.0, diameter=90.0, hub_height=80.0):
    farm = WindFarm(name="smoke")
    tid = 1
    turbines = []
    for r in range(rows):
        for c in range(cols):
            turbines.append(Turbine(id=f"T{tid:02d}", x=c * spacing, y=r * spacing,
                                     diameter=diameter, hub_height=hub_height))
            tid += 1
    farm.add_turbines(turbines)
    return farm


def compute_power(farm, wake_model, power_curve, speed=9.0, direction=270.0):
    farm.set_wind_condition(wind_speed=speed, wind_direction=direction)
    xi, eta = farm.wind_aligned_distances()
    total_power = 0.0
    for j, turbine in enumerate(farm.turbines):
        deficits = []
        for i, upstream in enumerate(farm.turbines):
            if xi[i, j] > 0:
                deficits.append(wake_model.wake_deficit(xi[i, j], eta[i, j], upstream.diameter))
        if deficits:
            u_eff = wake_model.effective_speed(speed, np.array(deficits))
        else:
            u_eff = speed
        total_power += power_curve.power(u_eff)
    return total_power


def main():
    farm = make_farm()
    sm = SimpleWakeModel(alpha=0.05, beta=0.5)
    jm = JensenWakeModel(wake_decay=0.05)
    em = EddyViscosityWakeModel(ambient_turbulence=0.10, thrust_coeff=0.8)
    pc = PowerCurve()

    print('SimpleWakeModel total power:', compute_power(farm, sm, pc))
    print('JensenWakeModel total power:', compute_power(farm, jm, pc))
    print('EddyViscosityWakeModel total power:', compute_power(farm, em, pc))


if __name__ == '__main__':
    main()
