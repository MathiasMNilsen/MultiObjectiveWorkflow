import numpy as np

def objectives(pred_data, input_dict, true_order):
    
    # Unpack some stuff
    kw_opt = input_dict
    report = true_order[1]
    wind_power_ens = np.load(kw_opt['windpower'])
    economic_conts = dict(kw_opt['npv_const'])

    # Define some variables
    ne = pred_data[0]['fopt'].shape[1]
    ndays = 0
    npv = 0.0
    co2 = [[] for _ in range(ne)]

    # Define a data getter
    get_data = lambda i, key: pred_data[i+1][key].squeeze() - pred_data[i][key].squeeze()
   
    for i in range(len(pred_data)-1):

        # Get volumes in interval
        oil_vol = get_data(i, 'fopt')
        gas_vol = get_data(i, 'fgpt')
        wp_vol  = get_data(i, 'fwpt')
        wi_vol  = get_data(i, 'fwit')
      
        # Calculate rates in interval
        idays = (report[i+1] - report[i]).days

        oil_rate = oil_vol/idays
        gas_rate = gas_vol/idays
        wp_rate  = wp_vol/idays
        wi_rate  = wi_vol/idays

        # Get maximum THP
        thp1 = (pred_data[i+1]['wthp a5'] + pred_data[i]['wthp a5'])/2
        thp2 = (pred_data[i+1]['wthp a6'] + pred_data[i]['wthp a6'])/2
        thp_max = np.maximum(thp1, thp2)

        # Get wind power
        wind_power = wind_power_ens[:, ndays:ndays+idays]
        
        # Row-stack rates
        oil_rate = np.row_stack([oil_rate]*idays).T
        gas_rate = np.row_stack([gas_rate]*idays).T
        wp_rate = np.row_stack([wp_rate]*idays).T
        wi_rate = np.row_stack([wi_rate]*idays).T
        thp_max = np.row_stack([thp_max]*idays).T

        # Calculate emissions
        co2_rate = []
        fuel_rate = []
        for n in range(ne):
            c, f = calc_emissions(oil_rate[n], gas_rate[n], wp_rate[n], wi_rate[n], thp_max[n], wind_power[n])
            co2_rate.append(c)
            fuel_rate.append(f)
        
        co2_rate = np.array(co2_rate)
        fuel_rate = np.array(fuel_rate)
        co2_vol = np.sum(co2_rate, axis=1)

        # Subtract fuel rate from gas production
        ton_to_sm3 = 1386
        gas_rate_exp = np.maximum(gas_rate - fuel_rate*ton_to_sm3, 0)
        gas_vol_exp  = np.sum(gas_rate_exp, axis=1)

        # Update ndays
        ndays += idays

        # calc NPV
        revenue = economic_conts['wop']*oil_vol + economic_conts['wgp']*gas_vol_exp
        expenditure = economic_conts['wwp']*wp_vol + economic_conts['wwi']*wi_vol + economic_conts['wem']*co2_vol
        dnpv = (revenue - expenditure)/(1 + economic_conts['disc'])**(ndays/365)
        npv += dnpv

        # Append emissions
        for n in range(ne):
            co2[n].extend(co2_rate[n].tolist())

    co2 = np.array(co2)    

    return npv, co2


from facility.gas_compressor import power_demand_gas_compressor
from facility.pump import power_demand_pump
from facility.water_treatment import power_demand_water_treatment
from facility.gas_turbine_system import turbine_system_consumption

def calc_emissions(oil_rate, gas_rate, wp_rate, wi_rate, thp_max, wind_power):

    pump_head = (thp_max-1)*10.199773339984054

    # Calculate power demand of components [MW]
    power_gas_comp  = power_demand_gas_compressor(gas_rate, P_max=22)
    power_wat_pump  = power_demand_pump(wi_rate, pump_head)
    power_wat_treat = power_demand_water_treatment(wp_rate)
    power_base_load = 4

    # Calculate total power demand [MW]
    total_power_demand = power_gas_comp + power_wat_pump + power_wat_treat + power_base_load

    # Calculate power load of gas turbines [MW]
    power_load_gas_turbines = np.maximum(total_power_demand - wind_power, 0)

    # calculate emissions and fuel rate [ton/day]
    emission_rate, fuel_rate = turbine_system_consumption(power_load_gas_turbines)

    return emission_rate, fuel_rate


