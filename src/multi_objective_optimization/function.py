import numpy as np

from popt.cost_functions.epf import epf

from facility.gas_compressor import power_demand_gas_compressor
from facility.pump import power_demand_pump
from facility.water_treatment import power_demand_water_treatment
from facility.gas_turbine_system import turbine_system_consumption

def objectives(pred_data, input_dict, true_order, **kwargs):

    # Unpack some stuff
    kw_opt = input_dict
    report = true_order[1]
    wind_power_ens = np.load(kw_opt['windpower'])
    economic_const = dict(kw_opt['npv_const'])

    # Define some variables
    ne = pred_data[0]['fopt'].shape[1]
    ndays = 0
    npv = 0.0
    co2 = [[] for _ in range(ne)]

    state = {}
    if 'wt' in economic_const:  # assume number of wind turbines and hydrogen storage are optimized
        state: dict = kwargs.get('state', {}) # state dictionary
        if state is {}:
            raise ValueError("Missing required input: x")
        if wind_power_ens.shape[0] < ne:
            raise ValueError("Wind power ensemble size is smaller than number of ensemble members")
        # Scale the wind power ensemble
        if state['N_WT'].size == 1:
            wind_power_ens[:ne] *= state['N_WT']
        else:
            wind_power_ens[:ne] *= np.squeeze(state['N_WT'])[:, np.newaxis]

    # Handle H2 storage if applicable
    epf_dict = kwargs.get('epf', {})
    r = -1
    if epf_dict:
        r = epf_dict.get('r', -1)  # epf penalty factor
        epf_dict['penalty'] = []  # initialize penalty list
    h2_storage = {}
    if 'h2' in economic_const:  # assume hydrogen storage is optimized
        h2_storage['h2cap'] = np.ones(ne)*np.squeeze(state['N_H2']) * 1.0e3  # max storage capacity in kg
        h2_storage['level'] = np.ones(ne)*np.squeeze(state['N_H2']) * 1.0e3  # initial capacity in kg (assume full)
        h2_storage['penalty'] = np.zeros(ne)  # initial penalty
        h2_storage['electrolysis'] = 50 / 1000  # Producing hydrogen via electrolysis typically requires about 50 kWh of
                                                # electricity per kilogram of hydrogen, assuming modern, efficient
                                                # electrolyzers (~70% efficiency).
        h2_storage['fuel_cell'] = 33.33 / 1000  # Using hydrogen in a fuel cell typically yields about 0.03333 MWh/kg
                                                # of H2 (Lower Heating Value (LHV): ~33.33 kWh/kg)
        h2_storage['eff'] = 0.5                 # assume 50% efficiency (TechnipFMC)

    # Define a data getter
    get_data = lambda i, key: pred_data[i+1][key].squeeze() - pred_data[i][key].squeeze()
    #data = [0.0] * 5  # store data for later analysis (if needed)
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
        thp1 = (pred_data[i+1]['wthp:a5'] + pred_data[i]['wthp:a5'])/2
        thp2 = (pred_data[i+1]['wthp:a6'] + pred_data[i]['wthp:a6'])/2
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
        fuel_rate = 0.0
        if 'h2' not in economic_const: # assume gas turbines are present
            co2_rate = []
            fuel_rate = []
            for n in range(ne):
                c, f = calc_emissions(oil_rate[n], gas_rate[n], wp_rate[n], wi_rate[n],
                                      thp_max[n], wind_power[n])
                co2_rate.append(c)
                fuel_rate.append(f)

            co2_rate = np.array(co2_rate)
            fuel_rate = np.array(fuel_rate)
            co2_vol = np.sum(co2_rate, axis=1)
        else:
            co2_rate = np.zeros((ne, idays))
            co2_vol = np.zeros(ne)

        if h2_storage:
            update_h2(oil_rate, gas_rate, wp_rate, wi_rate, thp_max, wind_power, h2_storage)

        # Subtract fuel rate from gas production
        ton_to_sm3 = 1386
        gas_rate_exp = np.maximum(gas_rate - fuel_rate*ton_to_sm3, 0)
        gas_vol_exp  = np.sum(gas_rate_exp, axis=1)

        # Update ndays
        ndays += idays

        # calc NPV
        revenue = economic_const['wop']*oil_vol + economic_const['wgp']*gas_vol_exp
        expenditure = economic_const['wwp']*wp_vol + economic_const['wwi']*wi_vol + economic_const['wem']*co2_vol
        dnpv = (revenue - expenditure)/(1 + float(economic_const['disc']))**(ndays/365)
        npv += dnpv

        capex = 0.0
        if h2_storage:
            capex += (economic_const['wt'] * np.squeeze(state['N_WT']) +
                      economic_const['h2'] * np.squeeze(state['N_H2']))

        # Append emissions
        for n in range(ne):
            co2[n].extend(co2_rate[n].tolist())

        # check for contraints
        if epf_dict and r >= 0:
            c_iq = h2_storage['penalty']
            penalty = epf(r, c_iq=c_iq[np.newaxis,:])
            epf_dict['penalty'].append(penalty)

        #data[0] += economic_const['wop']*oil_vol/(1 + float(economic_const['disc']))**(ndays/365)
        #data[1] += economic_const['wgp']*float(gas_vol_exp)/(1 + float(economic_const['disc']))**(ndays/365)
        #data[2] += economic_const['wwp']*wp_vol/(1 + float(economic_const['disc']))**(ndays/365)
        #data[3] += economic_const['wwi']*wi_vol/(1 + float(economic_const['disc']))**(ndays/365)
        #data[4] = capex

    # save npv data for later analysis
    #import pickle
    #with open('npv_data.pkl', 'wb') as f:
    #    pickle.dump(data, f)

    if epf_dict and epf_dict['penalty']:
        print(f'       -----> Mean EPF-Opt penalty term: {np.mean(np.concatenate(epf_dict['penalty']))}') # Print epf info
                
    co2 = np.array(co2)    

    return npv, co2, capex

def calc_emissions(oil_rate, gas_rate, wp_rate, wi_rate, thp_max, wind_power, **kwargs):

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
    emission_rate, fuel_rate = turbine_system_consumption(power_load_gas_turbines, **kwargs)

    return emission_rate, fuel_rate

def update_h2(oil_rate, gas_rate, wp_rate, wi_rate, thp_max, wind_power, h2_storage):

    ne = oil_rate.shape[0]
    for n in range(ne):
           
        pump_head = (thp_max[n]-1)*10.199773339984054
    
        # Calculate power demand of components [MW]
        power_gas_comp  = power_demand_gas_compressor(gas_rate[n], P_max=22)
        power_wat_pump  = power_demand_pump(wi_rate[n], pump_head)
        power_wat_treat = power_demand_water_treatment(wp_rate[n])
        power_base_load = 4
    
        # Calculate total power demand [MW]
        total_power_demand = power_gas_comp + power_wat_pump + power_wat_treat + power_base_load
    
        # Update global H2 storage level
        h2_storage['penalty'][n] = 0.0  # reset penalty
        for t in range(len(total_power_demand)):
            if total_power_demand[t] < wind_power[n][t]:
                excess_power = wind_power[n][t] - total_power_demand[t]
                excess_power *= 24  # Convert MW to MWh per day
                h2_diff = excess_power / h2_storage['electrolysis'] # convert to kilograms of H2
                h2_storage['level'][n] = min(h2_storage['level'][n] + h2_diff, h2_storage['h2cap'][n])  # max storage
            else:
                power_deficit = total_power_demand[t] - wind_power[n][t]
                power_deficit *= 24  # Convert MW to MWh per day
                h2_diff = power_deficit / h2_storage['fuel_cell']  # convert to kilograms of H2
                h2_diff /= h2_storage['eff'] # fuel cell efficiency 
                h2_storage['penalty'][n] += min(h2_storage['level'][n] - h2_diff, 0)  # penalty due to lack of H2
                h2_storage['level'][n] = max(h2_storage['level'][n] - h2_diff, 0)  # min storage 0 kg
                
               
                



