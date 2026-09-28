import numpy as np

# Constants
R  = 8.314     # J/(mol K)
Se = 15.4      # Energy content of natural gas [kWh/kg]

# Efficiency
polytropic_efficiency = lambda q: 0.49 + 0.03*np.log(q)

def power_demand_gas_compressor(rate, p_in=5, p_out=100, **kwargs):
    '''
    Source: https://myengineeringtools.com/Compressors/Polytropic_Compression.html#
    '''
    if rate == 0:
        return 0
    
    # unpack kwargs
    k = kwargs.get('k', 1.32)   # isentropic coefficient fo natural gas
    T = kwargs.get('T', 288.15) # Kelvin (15 Celcius)
    Z = kwargs.get('Z', 1)      # compressibility factor
    gas_molecular_weight = kwargs.get('molecular_weight', 19) # g/mol
    
    # calculate polytropic efficiency
    eff = polytropic_efficiency(rate)

    # define the polytropic coefficient of the gas
    n = 1/(1 -(k-1)/(k*eff))

    # converge rate from Sm3/day to kg/s
    kWh_per_Sm3 = 11.111 # kWh/Sm3
    rate = rate*kWh_per_Sm3/Se/(24*60*60) # kg/s

    # calculate the polytropic power
    Rprime = R/gas_molecular_weight
    power_polytropic = (n*Z*Rprime*T)/(n-1) * ( (p_out/p_in)**((n-1)/n) - 1) * rate
   
    power = power_polytropic/eff
    power = power/1000 # kW --> MWs
    return power

power_demand_gas_compressor = np.vectorize(power_demand_gas_compressor, otypes=[float])


