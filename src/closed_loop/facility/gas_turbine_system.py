import numpy as np

# constants
Sco2 = 2.75      # [kg CO2/kg fuel] Specific CO2 emission of gas fuel
Se = 0.0154      # [MWh/kg fuel] Specific energy content of gas fuel


def gas_trubine_efficiency(r):
    c0 = 0.001
    c1 = 3.166
    c2 = -5.723
    c3 = 7.218
    c4 = -5.106
    c5 = 1.445
    return c0 + c1*r + c2*r**2 + c3*r**3 + c4*r**4 + c5*r**5

def turbine_system_consumption(powerload, **kwargs):

    if powerload <= 0:
        return 0, 0


    # kwargs
    P_max = kwargs.get('P_max', 15)     # [MW] power output of one gas turbine at full load MW
    eff_max = kwargs.get('eff_max', 0.4)  # full load efficiency

    # coefficients for efficiency model
    c0 = 0.001
    c1 = 3.166
    c2 = -5.723
    c3 = 7.218
    c4 = -5.106
    c5 = 1.445
    
    # Number of Gas turbines
    n_gt = np.ceil(powerload/P_max)

    # power demand of each gas turbine
    P_pl = powerload/n_gt # [MW]

    # calculate efficiency at part load
    r = P_pl/P_max
    eff_pl = c0 + c1*r + c2*r**2 + c3*r**3 + c4*r**4 + c5*r**5

    # calculate overall efficiency
    eff = eff_max*eff_pl

    # calculate fuel mass rate
    fuel_mass_rate = n_gt*P_pl/(eff*Se) # kg/h
    fuel_mass_rate = fuel_mass_rate*24  # kg/day

    # calculate emission rate
    emission_rate = fuel_mass_rate*Sco2 # kg/day
    emission_rate = emission_rate/1000  # ton CO2/day
    return emission_rate, fuel_mass_rate/1000


turbine_system_consumption = np.vectorize(turbine_system_consumption)

