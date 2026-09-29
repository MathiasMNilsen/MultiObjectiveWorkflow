
def power_demand_water_treatment(rate):
    rate = rate/24/3600 # m3/s
    joule_per_Sm3 = 1.358e7
    power = rate*joule_per_Sm3
    
    # convert power from Joules to MW
    power = power/1e6
    return power