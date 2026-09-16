"""
energy_yield.py
================
Ties together the wind resource (wind_resource.py, already rescaled to hub
height via roughness if needed), the farm layout and wake model
(wind_farm.py / wake_model.py), and the turbine power curve (power_curve.py)
to produce:

  - a per-turbine table: x, y, rated power, gross energy, energy after the
    wake effect, then energy after each successive post-wake loss in the
    cascade (each one individually editable in the UI), down to net energy
  - farm totals, including the overall wake loss % and the net capacity
    factor Cf

Cf is defined as:

    Cf = total net energy [kWh/yr] / (sum of rated power [kW] * 8760 h)

i.e. actual net output divided by the output the farm would produce if
every turbine ran at rated power for every hour of the year.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

HOURS_PER_YEAR = 8760.0


@dataclass
class LossCascade:
    """Fractional (0-1) post-wake losses, applied multiplicatively in order.
    Each one is independently editable in the UI (no single combined %)."""
    shadow_flicker: float = 0.016
    turbine_flow_effect: float = 0.0108
    availability: float = 0.03
    electrical: float = 0.02
    turbine_performance: float = 0.036
    environment: float = 0.01

    def as_ordered_items(self):
        return [
            ("Shadow flicker", self.shadow_flicker),
            ("Turbine/flow effect", self.turbine_flow_effect),
            ("Availability", self.availability),
            ("Electrical", self.electrical),
            ("Turbine performance", self.turbine_performance),
            ("Environment", self.environment),
        ]

    def combined_factor(self) -> float:
        factor = 1.0
        for _, loss in self.as_ordered_items():
            factor *= (1.0 - loss)
        return factor


def compute_energy_table(farm, wake_model, power_curve, wind_rose,
                          losses: LossCascade, speed_bins=None, rated_power_kw=None):
    """
    Returns (rows: list[dict], totals: dict).
    All energies are in MWh/yr, powers in kW.

    losses : LossCascade
        Individually editable post-wake losses, applied multiplicatively
        in order after the wake effect has been computed.
    """
    if speed_bins is None:
        speed_bins = np.arange(0.5, 30.0, 1.0)  # 1 m/s wide bins

    n_sectors = wind_rose.n_sectors()
    n_turb = len(farm.turbines)

    gross_energy_per_turbine = np.zeros(n_turb)
    wake_energy_per_turbine = np.zeros(n_turb)

    bin_width = speed_bins[1] - speed_bins[0] if len(speed_bins) > 1 else 1.0
    gross_power_per_bin = np.array([power_curve.power(v) for v in speed_bins])  # kW

    for s_idx in range(n_sectors):
        direction = float(wind_rose.sector_centers_deg[s_idx])
        sector_freq = float(wind_rose.frequencies[s_idx])
        if sector_freq <= 0:
            continue

        pdf = wind_rose.weibull_pdf(speed_bins, s_idx)
        prob = pdf * bin_width * sector_freq  # probability mass per bin, this sector

        # Gross energy: free-stream, identical at every turbine location
        gross_hours = prob * HOURS_PER_YEAR
        gross_energy_mwh = float(np.sum(gross_hours * gross_power_per_bin) / 1000.0)
        gross_energy_per_turbine += gross_energy_mwh

        # Wake-affected energy needs this sector's direction-dependent geometry
        farm.set_wind_condition(wind_speed=max(float(np.mean(speed_bins)), 0.1),
                                wind_direction=direction)
        xi, eta = farm.wind_aligned_distances()

        for b_idx, v in enumerate(speed_bins):
            p_mass = prob[b_idx]
            if p_mass <= 0:
                continue
            hours = p_mass * HOURS_PER_YEAR
            for j, turbine in enumerate(farm.turbines):
                deficits = []
                for i, upstream in enumerate(farm.turbines):
                    if xi[i, j] > 0:
                        deficits.append(
                            wake_model.wake_deficit(xi[i, j], eta[i, j], upstream.diameter)
                        )
                u_eff = wake_model.effective_speed(v, np.array(deficits)) if deficits else v
                power_kw = power_curve.power(u_eff)
                wake_energy_per_turbine[j] += hours * power_kw / 1000.0

    rows = []
    total_gross = total_wake = total_net = total_rated_kw = 0.0
    combined_loss_factor = losses.combined_factor()

    for j, turbine in enumerate(farm.turbines):
        gross = float(gross_energy_per_turbine[j])
        after_wake = float(wake_energy_per_turbine[j])
        rated_kw = rated_power_kw if rated_power_kw is not None else power_curve.rated_power

        row = {
            "Turbine": turbine.id,
            "X [m]": turbine.x,
            "Y [m]": turbine.y,
            "Rated power [kW]": rated_kw,
            "Gross energy [MWh/yr]": gross,
            "After wake effect [MWh/yr]": after_wake,
            "Wake loss [%]": (1.0 - after_wake / gross) * 100.0 if gross > 0 else 0.0,
        }
        running = after_wake
        for name, loss in losses.as_ordered_items():
            running *= (1.0 - loss)
            row[f"After {name} [MWh/yr]"] = running
        row["Net energy [MWh/yr]"] = running

        rows.append(row)
        total_gross += gross
        total_wake += after_wake
        total_net += running
        total_rated_kw += rated_kw

    totals = {
        "Total rated power [kW]": total_rated_kw,
        "Total gross energy [MWh/yr]": total_gross,
        "Total energy after wake effect [MWh/yr]": total_wake,
        "Total net energy [MWh/yr]": total_net,
        "Total wake loss [%]": (1.0 - total_wake / total_gross) * 100.0 if total_gross > 0 else 0.0,
        "Combined post-wake loss factor": combined_loss_factor,
        "Combined post-wake loss [%]": (1.0 - combined_loss_factor) * 100.0,
        "Capacity factor Cf": (
            (total_net * 1000.0) / (total_rated_kw * HOURS_PER_YEAR)
            if total_rated_kw > 0 else 0.0
        ),
    }
    return rows, totals
