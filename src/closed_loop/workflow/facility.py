"""Facility power demand, gas-turbine emissions and interval economics.

The facility models are those of Nilsen (2025, Ch. 2.3). The facility has four power consumers: a
compressor train for the produced gas, pumps for water injection, water treatment and a constant base load.
Gas turbines cover whatever the wind power cannot. The economics follow Eqs. (2)-(3) of the paper, without
discounting inside an interval.

All functions work on daily arrays. Rates are in Sm3/day, power in MW and emissions in ton/day.
"""
import numpy as np

BASE_LOAD = 3.0                  # constant power demand [MW]
TURBINE_POWER = 15.0             # rated power of one gas turbine [MW]
GAS_DENSITY = 0.7215             # fuel-gas density [kg/Sm3]
M_PER_BAR = 10.199773339984054   # water head per pressure [m/bar]


def evaluate(response, wind, economics):
    """Power demand and emissions per day, plus the interval totals and the undiscounted NPV.

    `response` maps summary keys such as 'WOPR:A1' to daily rates, and `wind` is the available wind
    power per day [MW]. `wind` may also hold scenarios (n_scenarios x n_days). The daily emissions then
    have a scenario axis, and the CO2 and NPV totals are averaged over the scenarios.
    """
    def field(vector):
        return np.sum([rate for key, rate in response.items() if key.startswith(vector + ":")], axis=0)

    oil, gas, water, injection = field("WOPR"), field("WGPR"), field("WWPR"), field("WWIR")
    thp = np.max([p for key, p in response.items() if key.startswith("WTHP:")], axis=0)
    power = (compressor_power(gas) + pump_power(injection, (thp - 1) * M_PER_BAR)
             + water_treatment_power(water) + BASE_LOAD)
    co2_rate, fuel_rate = turbine_consumption(np.maximum(power - wind, 0))
    gas_export = np.maximum(gas - fuel_rate * 1e3 / GAS_DENSITY, 0)  # the turbines burn produced gas

    totals = {"oil": oil.sum(), "gas": gas.sum(), "wp": water.sum(), "wi": injection.sum(),
              "co2": co2_rate.sum(axis=-1).mean()}
    totals["npv"] = (economics["oil_price"] * totals["oil"] + economics["gas_price"] * gas_export.sum(axis=-1).mean()
                     - economics["water_injection_cost"] * totals["wi"]
                     - economics["water_production_cost"] * totals["wp"]
                     - economics["co2_tax"] * totals["co2"])
    return {"power": power, "co2_rate": co2_rate, **totals}


def compressor_power(rate):
    """Polytropic compression of the produced gas from 5 to 100 bar."""
    k, T, Z, R = 1.32, 288.15, 1.0, 8.314 / 19  # isentropic exponent, K, -, kJ/(kg K) for 19 g/mol
    on = rate > 0
    rate = np.where(on, rate, 1.0)
    eff = 0.49 + 0.03 * np.log(rate)            # polytropic efficiency
    n = 1 / (1 - (k - 1) / (k * eff))           # polytropic exponent
    mass_rate = rate * 11.111 / 15.4 / 86400    # kg/s (11.111 kWh/Sm3 and 15.4 kWh/kg)
    power = (n * Z * R * T) / (n - 1) * ((100 / 5) ** ((n - 1) / n) - 1) * mass_rate / eff / 1000
    return np.where(on, power, 0.0)


def pump_power(rate, head):
    """Water-injection pumps: the cheapest of 1-10 parallel x 1-10 serial pumps that delivers `head` [m]."""
    n_parallel = np.arange(1, 11)[:, None, None]
    n_series = np.arange(1, 11)[None, :, None]
    q = rate / n_parallel                                      # rate per pump [Sm3/day]
    system_head = _pump_head(q) * n_series
    power = (9.81 * 1000) * system_head * (rate / 24 / 3600) / _pump_efficiency(q) / 1e6
    power = np.where(system_head >= head, power, np.inf).min(axis=(0, 1))
    return np.where((rate > 1e-8) & (head > 1e-8), power, 0.0)


def _pump_head(q, omega=2500):
    """Head [m] of one pump at rate q [Sm3/day] and speed omega [rpm]."""
    q = q / 24 / 3600
    return (1.823e3 - 56.41 * (q - 0.1593) / 0.08964
            + 64.09 * (omega * q - 7.068e2) / 4.196e2
            + 2.061e2 * (omega**2 - 1.942e7) / 3.681e6
            + 1.712e2 * (omega**3 - 8.672e10) / 2.407e10
            - 1.868e2 * (q**3 - 7.922e-3) / 9.304e-3)


def _pump_efficiency(q, omega=2500):
    """Efficiency of one pump at rate q [Sm3/day] and speed omega [rpm]."""
    q = q / 24 / 3600
    return 0.95 * (1.000e-10 + 9.641 * q - 38.12 * q**2 + 5.741e-3 * omega * q**2 - 1.629e-7 * q * omega**2)


def water_treatment_power(rate):
    """Treatment of the produced water (1.358e7 J/Sm3)."""
    return rate / 24 / 3600 * 1.358e7 / 1e6


def turbine_consumption(load):
    """CO2 emission and fuel consumption [ton/day] of the gas turbines covering `load` [MW]."""
    on = load > 0
    load = np.where(on, load, 1.0)
    n = np.ceil(load / TURBINE_POWER)                 # turbines in operation, sharing the load
    r = load / n / TURBINE_POWER                      # part-load ratio
    eff = 0.4 * (0.001 + 3.166 * r - 5.723 * r**2 + 7.218 * r**3 - 5.106 * r**4 + 1.445 * r**5)
    fuel = np.where(on, n * (load / n) / (eff * 0.0154) * 24, 0.0)  # kg/day (0.0154 MWh/kg)
    return fuel * 2.75 / 1000, fuel / 1000                          # 2.75 kg CO2 per kg fuel
