"""Tests for the wind-farm energy model.

They check physics and bookkeeping that can be verified by hand: the Jensen
wake formula, the power curve, Weibull statistics, log-law height scaling,
the loss cascade, and the farm-level energy table.
"""
import math

import numpy as np
import pytest
from scipy.special import gamma

from energy_yield import LossCascade, compute_energy_table
from power_curve import PowerCurve
from turbine import Turbine
from wake_model import EddyViscosityWakeModel, JensenWakeModel
from wind_farm import WindFarm
from wind_resource import WindRose

D = 54.0


def make_farm(rows=2, cols=3, spacing_d=5.0, diameter=D):
    farm = WindFarm(name="test")
    for r in range(rows):
        for c in range(cols):
            farm.add_turbine(
                Turbine(
                    id=f"T{r * cols + c + 1:02d}",
                    x=c * spacing_d * diameter,
                    y=r * spacing_d * diameter,
                    diameter=diameter,
                    hub_height=50.0,
                )
            )
    return farm


def single_direction_rose(direction=270.0, a=9.0, k=2.0):
    centers = np.arange(0.0, 360.0, 30.0)
    freqs = np.where(centers == direction, 1.0, 0.0)
    return WindRose(centers, freqs, np.full(12, k), np.full(12, a))


# --- Jensen wake model -------------------------------------------------------

def test_jensen_matches_hand_calculation():
    k, ct, x = 0.075, 0.8, 5 * D
    model = JensenWakeModel(wake_decay=k, thrust_coeff=ct)
    r0 = D / 2
    r = r0 + k * x
    expected = (1 - math.sqrt(1 - ct)) * (r0 / r) ** 2
    assert model.wake_deficit(x, 0.0, D) == pytest.approx(expected)


def test_jensen_has_no_deficit_upstream_or_outside_the_wake():
    model = JensenWakeModel(wake_decay=0.075)
    assert model.wake_deficit(-100.0, 0.0, D) == 0.0
    assert model.wake_deficit(0.0, 0.0, D) == 0.0
    x = 5 * D
    wake_radius = D / 2 + 0.075 * x
    assert model.wake_deficit(x, wake_radius * 1.01, D) == 0.0
    assert model.wake_deficit(x, wake_radius * 0.99, D) > 0.0


def test_jensen_deficit_decays_with_distance():
    model = JensenWakeModel(wake_decay=0.075)
    deficits = [model.wake_deficit(n * D, 0.0, D) for n in (2, 4, 6, 8, 10)]
    assert all(a > b for a, b in zip(deficits, deficits[1:]))
    assert all(0 < d < 1 for d in deficits)


def test_deficit_arrays_work_like_scalars():
    model = JensenWakeModel(wake_decay=0.075)
    xi = np.array([2 * D, 4 * D, -D])
    eta = np.zeros(3)
    out = model.wake_deficit(xi, eta, D)
    assert out.shape == (3,)
    assert out[0] == pytest.approx(model.wake_deficit(2 * D, 0.0, D))
    assert out[2] == 0.0


def test_effective_speed_sums_deficits_and_caps_the_total():
    model = JensenWakeModel()
    assert model.effective_speed(10.0, np.array([0.1, 0.1])) == pytest.approx(8.0)
    assert model.effective_speed(10.0, np.array([0.7, 0.7])) == pytest.approx(0.1)


def test_eddy_viscosity_wake_decays_and_stays_physical():
    model = EddyViscosityWakeModel(ambient_turbulence=0.10, thrust_coeff=0.8)
    near = model.wake_deficit(4 * D, 0.0, D)
    far = model.wake_deficit(10 * D, 0.0, D)
    assert 0 < far < near < 1
    assert model.wake_deficit(-D, 0.0, D) == 0.0
    assert model.wake_deficit(4 * D, 10 * D, D) < near * 0.01


# --- Power curve -------------------------------------------------------------

def test_power_curve_regions():
    pc = PowerCurve(cut_in=3.0, rated_speed=12.0, cut_out=25.0, rated_power=1000.0)
    assert pc.power(2.9) == 0.0
    assert pc.power(3.0) == 0.0
    assert pc.power(12.0) == pytest.approx(1000.0)
    assert pc.power(20.0) == pytest.approx(1000.0)
    assert pc.power(25.1) == 0.0


def test_power_curve_ramp_is_cubic_and_monotonic():
    pc = PowerCurve(cut_in=3.0, rated_speed=12.0, cut_out=25.0, rated_power=1000.0)
    assert pc.power(7.5) == pytest.approx(1000.0 * (4.5 / 9.0) ** 3)
    speeds = np.linspace(3.0, 12.0, 25)
    powers = [pc.power(v) for v in speeds]
    assert all(a <= b for a, b in zip(powers, powers[1:]))


def test_tabulated_power_curve_interpolates_and_is_zero_outside():
    pc = PowerCurve(speed_table=[4, 8, 12], power_table=[0, 500, 1000])
    assert pc.power(6.0) == pytest.approx(250.0)
    assert pc.power(3.0) == 0.0
    assert pc.power(13.0) == 0.0
    assert pc.rated_power == 1000.0


# --- Wind resource -----------------------------------------------------------

def test_weibull_mean_speed_matches_gamma_function():
    rose = WindRose([0.0], [1.0], [2.0], [10.0])
    assert rose.mean_speed(0) == pytest.approx(10.0 * gamma(1.5))


def test_weibull_pdf_integrates_to_one():
    rose = WindRose([0.0], [1.0], [2.2], [8.0])
    speeds = np.linspace(0.0, 60.0, 6001)
    pdf = rose.weibull_pdf(speeds, 0)
    area = float(np.sum((pdf[1:] + pdf[:-1]) / 2.0 * np.diff(speeds)))
    assert area == pytest.approx(1.0, abs=1e-3)


def test_wind_rose_frequencies_are_normalised():
    rose = WindRose([0.0, 180.0], [3.0, 1.0], [2.0, 2.0], [8.0, 8.0])
    assert rose.frequencies.sum() == pytest.approx(1.0)
    assert rose.frequencies[0] == pytest.approx(0.75)


def test_wind_rose_rejects_empty_frequencies():
    with pytest.raises(ValueError):
        WindRose([0.0], [0.0], [2.0], [8.0])


def test_height_rescaling_follows_the_log_law():
    rose = WindRose([0.0], [1.0], [2.0], [5.0])
    scaled = rose.rescaled_to_height(target_height=50.0, roughness_z0=0.03)
    factor = math.log(50.0 / 0.03) / math.log(10.0 / 0.03)
    assert scaled.weibull_a[0] == pytest.approx(5.0 * factor)
    assert scaled.weibull_k[0] == rose.weibull_k[0]
    assert scaled.overall_mean_speed() > rose.overall_mean_speed()


def test_height_rescaling_rejects_non_positive_roughness():
    rose = WindRose([0.0], [1.0], [2.0], [5.0])
    with pytest.raises(ValueError):
        rose.rescaled_to_height(50.0, 0.0)


# --- Farm geometry -----------------------------------------------------------

def test_wind_aligned_distances_for_a_westerly_wind():
    farm = WindFarm()
    farm.add_turbine(Turbine("A", 0.0, 0.0, D, 50.0))
    farm.add_turbine(Turbine("B", 450.0, 0.0, D, 50.0))
    farm.set_wind_condition(9.0, 270.0)
    xi, eta = farm.wind_aligned_distances()
    assert xi[0, 1] == pytest.approx(450.0)
    assert xi[1, 0] == pytest.approx(-450.0)
    assert abs(eta[0, 1]) < 1e-9


def test_farm_rejects_duplicate_ids_and_bad_inputs():
    farm = WindFarm()
    farm.add_turbine(Turbine("A", 0.0, 0.0, D, 50.0))
    with pytest.raises(ValueError):
        farm.add_turbine(Turbine("A", 100.0, 0.0, D, 50.0))
    with pytest.raises(ValueError):
        farm.set_wind_condition(0.0, 270.0)
    with pytest.raises(ValueError):
        Turbine("B", 0.0, 0.0, -1.0, 50.0)


# --- Losses and energy -------------------------------------------------------

def test_loss_cascade_is_multiplicative():
    losses = LossCascade()
    expected = 1.0
    for _, loss in losses.as_ordered_items():
        expected *= 1.0 - loss
    assert losses.combined_factor() == pytest.approx(expected)
    assert 0.88 < losses.combined_factor() < 0.89


def test_isolated_turbine_has_no_wake_loss():
    farm = make_farm(rows=1, cols=1)
    rows, totals = compute_energy_table(
        farm, JensenWakeModel(0.075), PowerCurve(), WindRose.default_12_sector(), LossCascade()
    )
    assert totals["Total wake loss [%]"] == pytest.approx(0.0, abs=1e-9)
    assert totals["Total net energy [MWh/yr]"] == pytest.approx(
        totals["Total gross energy [MWh/yr]"] * LossCascade().combined_factor()
    )
    assert 0.0 < totals["Capacity factor Cf"] < 1.0


def test_energy_scales_with_wind_speed():
    farm = make_farm(rows=1, cols=1)
    args = (farm, JensenWakeModel(0.075), PowerCurve())
    slow = compute_energy_table(*args, WindRose.default_12_sector(prevailing_speed=6.0), LossCascade())[1]
    fast = compute_energy_table(*args, WindRose.default_12_sector(prevailing_speed=10.0), LossCascade())[1]
    assert fast["Total gross energy [MWh/yr]"] > slow["Total gross energy [MWh/yr]"] * 1.5


def test_downstream_turbine_loses_energy_to_the_wake():
    farm = WindFarm()
    farm.add_turbine(Turbine("front", 0.0, 0.0, D, 50.0))
    farm.add_turbine(Turbine("back", 5 * D, 0.0, D, 50.0))
    rows, totals = compute_energy_table(
        farm, JensenWakeModel(0.075), PowerCurve(), single_direction_rose(270.0), LossCascade()
    )
    front, back = rows
    assert front["Wake loss [%]"] == pytest.approx(0.0, abs=1e-9)
    assert back["Wake loss [%]"] > 5.0
    assert totals["Total wake loss [%]"] > 0.0


def test_one_wind_direction_overstates_wake_loss_compared_with_a_spread_rose():
    args = (JensenWakeModel(0.075), PowerCurve())
    single = compute_energy_table(make_farm(), *args, single_direction_rose(270.0), LossCascade())[1]
    spread = compute_energy_table(make_farm(), *args, WindRose.default_12_sector(), LossCascade())[1]
    assert single["Total wake loss [%]"] > spread["Total wake loss [%]"]
