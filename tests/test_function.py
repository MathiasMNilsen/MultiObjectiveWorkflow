import datetime as dt

import numpy as np
import pytest

from facility.gas_compressor import power_demand_gas_compressor
from facility.gas_turbine_system import turbine_system_consumption
from facility.pump import power_demand_pump
from facility.water_treatment import power_demand_water_treatment
from multi_objective_optimization.function import calc_emissions, objectives, update_h2

NE = 2
REPORT = [dt.datetime(2020, 7, 1), dt.datetime(2020, 8, 1), dt.datetime(2020, 9, 1)]
NDAYS = (REPORT[-1] - REPORT[0]).days
RATES = {'fopt': 2000.0, 'fgpt': 3.0e5, 'fwpt': 1500.0, 'fwit': 8000.0}  # Sm3/day
THP = 50.0  # bar
GAS_CONST = {'wop': 471.7, 'wgp': 0.4, 'wwp': 37, 'wwi': 25, 'wem': 150, 'disc': 0.08}


def make_pred_data(rates=RATES, thp=THP, ne=NE):
    """Cumulative totals growing at constant daily rates, one dict per report step."""
    pred_data = []
    for date in REPORT:
        days = (date - REPORT[0]).days
        step = {key: np.full((1, ne), rate * days) for key, rate in rates.items()}
        step['wthp:a5'] = np.full((1, ne), thp)
        step['wthp:a6'] = np.full((1, ne), thp - 5.0)
        pred_data.append(step)
    return pred_data


def make_input(tmp_path, wind_level, const=GAS_CONST, ne=NE):
    path = tmp_path / 'wp.npy'
    np.save(path, np.full((ne + 1, NDAYS), float(wind_level)))
    return {'windpower': str(path), 'npv_const': dict(const)}


def expected_npv_without_emissions(const=GAS_CONST):
    npv, ndays = 0.0, 0
    for start, end in zip(REPORT[:-1], REPORT[1:]):
        days = (end - start).days
        ndays += days
        cash = (const['wop'] * RATES['fopt'] + const['wgp'] * RATES['fgpt']
                - const['wwp'] * RATES['fwpt'] - const['wwi'] * RATES['fwit']) * days
        npv += cash / (1 + const['disc']) ** (ndays / 365)
    return npv


def total_power_demand(rates=RATES, thp=THP):
    head = (thp - 1) * 10.199773339984054
    return (power_demand_gas_compressor(rates['fgpt'], P_max=22)
            + power_demand_pump(rates['fwit'], head)[0]
            + power_demand_water_treatment(rates['fwpt']) + 4)


# ---------------------------------------------------------------- calc_emissions

def test_calc_emissions_zero_when_wind_covers_demand():
    days = 5
    ones = np.ones(days)
    co2, fuel = calc_emissions(ones * 2000, ones * 3e5, ones * 1500, ones * 8000,
                               ones * THP, ones * 1e3)
    np.testing.assert_array_equal(co2, 0)
    np.testing.assert_array_equal(fuel, 0)


def test_calc_emissions_without_wind_matches_turbine_model():
    days = 4
    ones = np.ones(days)
    co2, fuel = calc_emissions(ones * RATES['fopt'], ones * RATES['fgpt'], ones * RATES['fwpt'],
                               ones * RATES['fwit'], ones * THP, np.zeros(days))
    exp_co2, exp_fuel = turbine_system_consumption(total_power_demand())
    np.testing.assert_allclose(co2, exp_co2)
    np.testing.assert_allclose(fuel, exp_fuel)


def test_calc_emissions_wind_reduces_emissions():
    days = 3
    ones = np.ones(days)
    args = (ones * RATES['fopt'], ones * RATES['fgpt'], ones * RATES['fwpt'],
            ones * RATES['fwit'], ones * THP)
    co2_no_wind, _ = calc_emissions(*args, np.zeros(days))
    co2_wind, _ = calc_emissions(*args, ones * 0.5 * total_power_demand())
    assert np.all(co2_wind < co2_no_wind)


# ---------------------------------------------------------------- objectives: gas case

def test_objectives_gas_output_shapes(tmp_path):
    npv, co2, capex = objectives(make_pred_data(), make_input(tmp_path, 0.0), (None, REPORT))
    assert np.shape(npv) == (NE,)
    assert co2.shape == (NE, NDAYS)
    assert capex == 0.0


def test_objectives_gas_full_wind_matches_hand_computed_npv(tmp_path):
    npv, co2, _ = objectives(make_pred_data(), make_input(tmp_path, 1e3), (None, REPORT))
    np.testing.assert_array_equal(co2, 0.0)
    np.testing.assert_allclose(npv, expected_npv_without_emissions(), rtol=1e-12)


def test_objectives_gas_no_wind_emits_and_lowers_npv(tmp_path):
    npv_wind, _, _ = objectives(make_pred_data(), make_input(tmp_path, 1e3), (None, REPORT))
    npv_gas, co2, _ = objectives(make_pred_data(), make_input(tmp_path, 0.0), (None, REPORT))
    assert np.all(co2 > 0)
    assert np.all(npv_gas < npv_wind)


def test_objectives_gas_daily_co2_matches_calc_emissions(tmp_path):
    _, co2, _ = objectives(make_pred_data(), make_input(tmp_path, 0.0), (None, REPORT))
    expected, _ = turbine_system_consumption(total_power_demand())
    np.testing.assert_allclose(co2, expected)


def test_objectives_emission_cost_scales_npv_gap(tmp_path):
    const_cheap = dict(GAS_CONST, wem=0.0)
    npv_free, _, _ = objectives(make_pred_data(), make_input(tmp_path, 0.0, const_cheap), (None, REPORT))
    npv_taxed, _, _ = objectives(make_pred_data(), make_input(tmp_path, 0.0), (None, REPORT))
    assert np.all(npv_taxed < npv_free)


# ---------------------------------------------------------------- objectives: hydrogen case

H2_CONST = dict(GAS_CONST, wt=1.0e6, h2=2.0e3)


def h2_state(n_wt=2.0, n_h2=100.0):
    return {'N_WT': np.array([n_wt]), 'N_H2': np.array(n_h2)}


def test_objectives_hydrogen_has_no_emissions_and_capex(tmp_path):
    npv, co2, capex = objectives(make_pred_data(), make_input(tmp_path, 0.0, H2_CONST),
                                 (None, REPORT), state=h2_state())
    np.testing.assert_array_equal(co2, 0.0)
    assert capex == pytest.approx(1.0e6 * 2.0 + 2.0e3 * 100.0)
    # No gas turbines -> no fuel use and no emission cost, so NPV equals the emission-free case.
    np.testing.assert_allclose(npv, expected_npv_without_emissions(), rtol=1e-12)


def test_objectives_hydrogen_per_member_turbines(tmp_path):
    state = {'N_WT': np.array([1.0, 3.0]), 'N_H2': np.array(10.0)}
    npv, _, capex = objectives(make_pred_data(), make_input(tmp_path, 1.0, H2_CONST),
                               (None, REPORT), state=state)
    np.testing.assert_allclose(capex, 1.0e6 * np.array([1.0, 3.0]) + 2.0e3 * 10.0)
    assert np.shape(npv) == (NE,)


def test_objectives_hydrogen_small_wind_ensemble_raises(tmp_path):
    path = tmp_path / 'wp.npy'
    np.save(path, np.zeros((1, NDAYS)))
    with pytest.raises(ValueError, match='Wind power ensemble size'):
        objectives(make_pred_data(), {'windpower': str(path), 'npv_const': H2_CONST},
                   (None, REPORT), state=h2_state())


def test_objectives_hydrogen_epf_penalty_without_wind(tmp_path):
    epf = {'r': 1.0}
    objectives(make_pred_data(), make_input(tmp_path, 0.0, H2_CONST), (None, REPORT),
               state=h2_state(n_h2=1.0), epf=epf)
    assert len(epf['penalty']) == len(REPORT) - 1
    assert all(np.all(p > 0) for p in epf['penalty'])


def test_objectives_hydrogen_epf_no_penalty_with_ample_wind(tmp_path):
    epf = {'r': 1.0}
    objectives(make_pred_data(), make_input(tmp_path, 1e3, H2_CONST), (None, REPORT),
               state=h2_state(n_wt=1.0), epf=epf)
    assert all(np.all(p == 0) for p in epf['penalty'])


def test_objectives_negative_epf_factor_skips_penalty(tmp_path):
    epf = {'r': -1}
    objectives(make_pred_data(), make_input(tmp_path, 0.0, H2_CONST), (None, REPORT),
               state=h2_state(n_h2=1.0), epf=epf)
    assert epf['penalty'] == []


# ---------------------------------------------------------------- update_h2

def make_storage(capacity_kg, level_kg, ne=1):
    return {
        'h2cap': np.full(ne, capacity_kg),
        'level': np.full(ne, level_kg),
        'penalty': np.zeros(ne),
        'electrolysis': 50 / 1000,
        'fuel_cell': 33.33 / 1000,
        'eff': 0.5,
    }


def rate_rows(days):
    ones = np.ones((1, days))
    return (ones * RATES['fopt'], ones * RATES['fgpt'], ones * RATES['fwpt'],
            ones * RATES['fwit'], ones * THP)


def test_update_h2_excess_wind_fills_storage_to_capacity():
    storage = make_storage(1000.0, 0.0)
    update_h2(*rate_rows(3), np.full((1, 3), 1e3), storage)
    assert storage['level'][0] == pytest.approx(1000.0)
    assert storage['penalty'][0] == 0.0


def test_update_h2_deficit_drains_storage():
    demand = total_power_demand()
    storage = make_storage(1e7, 1e7)
    update_h2(*rate_rows(1), np.zeros((1, 1)), storage)
    used = demand * 24 / (33.33 / 1000) / 0.5
    assert storage['level'][0] == pytest.approx(1e7 - used)
    assert storage['penalty'][0] == 0.0


def test_update_h2_empty_storage_gives_negative_penalty():
    storage = make_storage(10.0, 10.0)
    update_h2(*rate_rows(2), np.zeros((1, 2)), storage)
    assert storage['level'][0] == 0.0
    assert storage['penalty'][0] < 0.0
