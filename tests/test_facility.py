import numpy as np
import pytest

from facility.gas_compressor import power_demand_gas_compressor, polytropic_efficiency
from facility.gas_turbine_system import (Sco2, gas_trubine_efficiency,
                                         turbine_system_consumption)
from facility.pump import (power_demand_pump, pump_eff, pump_head, system_head,
                           system_power_demand)
from facility.water_treatment import power_demand_water_treatment


# ---------------------------------------------------------------- water treatment

def test_water_treatment_is_linear_in_rate():
    assert power_demand_water_treatment(0.0) == 0.0
    p1 = power_demand_water_treatment(1000.0)
    p2 = power_demand_water_treatment(2000.0)
    assert p2 == pytest.approx(2 * p1)


def test_water_treatment_known_value():
    # 86400 Sm3/day = 1 m3/s -> 1.358e7 J/s = 13.58 MW
    assert power_demand_water_treatment(86400.0) == pytest.approx(13.58)


def test_water_treatment_accepts_arrays():
    rates = np.array([0.0, 86400.0])
    np.testing.assert_allclose(power_demand_water_treatment(rates), [0.0, 13.58])


# ---------------------------------------------------------------- gas compressor

def test_compressor_zero_rate_needs_no_power():
    assert power_demand_gas_compressor(0) == 0


def test_compressor_power_positive_and_increasing():
    rates = np.array([1e5, 5e5, 1e6, 2e6])
    power = power_demand_gas_compressor(rates)
    assert power.dtype.kind == 'f'
    assert np.all(power > 0)
    assert np.all(np.diff(power) > 0)


def test_compressor_higher_pressure_ratio_needs_more_power():
    low = power_demand_gas_compressor(1e6, p_in=5, p_out=50)
    high = power_demand_gas_compressor(1e6, p_in=5, p_out=100)
    assert high > low


def test_compressor_efficiency_grows_with_rate():
    assert polytropic_efficiency(1e6) > polytropic_efficiency(1e5)


def test_compressor_leading_zero_does_not_truncate():
    power = power_demand_gas_compressor(np.array([0.0, 1e5, 1e6]))
    np.testing.assert_allclose(power[1:], power_demand_gas_compressor(np.array([1e5, 1e6])))


# ---------------------------------------------------------------- gas turbines

def test_turbine_zero_load_no_emissions():
    emission, fuel = turbine_system_consumption(0)
    assert emission == 0 and fuel == 0


def test_turbine_efficiency_is_one_at_full_load():
    assert gas_trubine_efficiency(1.0) == pytest.approx(1.001)


def test_turbine_emission_is_fuel_times_specific_co2():
    loads = np.array([5.0, 15.0, 22.0, 40.0])
    emission, fuel = turbine_system_consumption(loads)
    np.testing.assert_allclose(emission, fuel * Sco2)


def test_turbine_full_load_single_unit():
    # 15 MW at full load: fuel = 15 / (0.4 * 1.001 * 0.0154) kg/h -> tonne/day
    emission, fuel = turbine_system_consumption(15.0)
    expected_fuel = 15.0 / (0.4 * 1.001 * 0.0154) * 24 / 1000
    assert fuel == pytest.approx(expected_fuel)
    assert emission == pytest.approx(expected_fuel * Sco2)


def test_turbine_fuel_increases_with_load():
    loads = np.linspace(1.0, 45.0, 20)
    _, fuel = turbine_system_consumption(loads)
    assert np.all(fuel > 0)
    # Part-load efficiency drops, but total fuel should still grow with the delivered power overall.
    assert fuel[-1] > fuel[0]


def test_turbine_custom_rating():
    _, fuel_default = turbine_system_consumption(20.0)
    _, fuel_big = turbine_system_consumption(20.0, P_max=25.0)
    assert fuel_default != pytest.approx(fuel_big)


def test_turbine_leading_zero_does_not_truncate():
    emission, _ = turbine_system_consumption(np.array([0.0, 10.0]))
    assert emission[1] == pytest.approx(turbine_system_consumption(10.0)[0])


# ---------------------------------------------------------------- pump

def test_pump_scalar_input_returns_single_value_list():
    power = power_demand_pump(5000.0, 100.0)
    assert isinstance(power, list) and len(power) == 1
    assert power[0] > 0


def test_pump_array_input():
    power = power_demand_pump([5000.0, 10000.0], [100.0, 300.0])
    assert len(power) == 2
    assert all(p > 0 for p in power)


def test_pump_scalar_head_broadcasts():
    power = power_demand_pump([5000.0, 10000.0], 100.0)
    assert len(power) == 2


def test_pump_mismatched_lengths_raise():
    with pytest.raises(ValueError, match='same length'):
        power_demand_pump([5000.0, 6000.0], [100.0, 200.0, 300.0])


def test_pump_unreachable_head_raises():
    # No parallel/series configuration can deliver this head.
    with pytest.raises(ValueError):
        power_demand_pump(5000.0, 1e6)


def test_pump_higher_head_needs_at_least_as_much_power():
    low = power_demand_pump(8000.0, 100.0)[0]
    high = power_demand_pump(8000.0, 2000.0)[0]
    assert high >= low


def test_pump_chooses_cheapest_feasible_configuration():
    Q, H = 8000.0, 500.0
    feasible = [system_power_demand(Q, p, s)
                for p in range(1, 6) for s in range(1, 6)
                if system_head(Q, p, s) >= H]
    assert power_demand_pump(Q, H)[0] == pytest.approx(min(feasible))


def test_system_head_scales_with_series_pumps():
    assert system_head(5000.0, 1, 3) == pytest.approx(3 * pump_head(5000.0))
    assert system_head(5000.0, 2, 1) == pytest.approx(pump_head(2500.0))


def test_pump_efficiency_in_physical_range():
    for q in [1000.0, 5000.0, 10000.0]:
        assert 0.0 < pump_eff(q) < 1.0
