import numpy  as np
import pandas as pd
import os
import shutil

from misc import ecl
from glob import glob
from mako.template import Template
from resdata.summary import Summary
from p_tqdm import p_map

# Internal
try:
    from ..facility.gas_compressor import power_demand_gas_compressor
    from ..facility.gas_turbine_system import turbine_system_consumption
    from ..facility.pump import power_demand_pump
    from ..facility.water_treatment import power_demand_water_treatment

except Exception:
    # fallback when run as a script from the project root
    from facility.gas_compressor import power_demand_gas_compressor
    from facility.gas_turbine_system import turbine_system_consumption
    from facility.pump import power_demand_pump
    from facility.water_treatment import power_demand_water_treatment


def simulate(filename, input, dates, rst_idx, input_keys=None, output_keys=None, njobs=1, **kwargs):
    
    # Check if input is array, list of dataframe or just dataframe
    if isinstance(input, np.ndarray):
        x = input
        if input_keys is None:
            raise ValueError('input_keys must be provided when input is ndarray')
        output_type = 'array'

    elif isinstance(input, list):
        x = np.array([dataframe_to_vec(df) for df in input])
        output_type = 'list'

    elif isinstance(input, pd.DataFrame):
        input_keys = list(input.columns)
        x = dataframe_to_vec(input, input_keys)
        output_type = 'dataframe'

    # Check shape
    if len(x.shape) > 1:
        ne, nx = x.shape
    else:
        ne = 1
        nx = x.shape[0]
        x = np.array([x]) 

    # Prepare tasks for each ensemble member
    tasks = []
    for n in range(ne):
        df = vec_to_dataframe(x[n], input_keys, dates)
        df_dict = df.to_dict(orient='list')
        simfolder = f'Sim{n}'
        tasks.append((filename, simfolder, df_dict))


    # Define runner function
    def _runner(args):
        fn, simfolder_, df_d = args
        status = simulate_mako(fn, simfolder_, **df_d, rep_dates=dates, rst_index=rst_idx)

        if status == 0:
            # Get sim results
            res = get_sim_results(
                casename=f'{simfolder_}/{fn}',     
                datatypes=output_keys, 
                dates=dates
            )

            # Get restart file
            if kwargs.get('get_restart', False):
                get_latest_restart_index(casename=fn, folder=simfolder_)
        else:
            res = None

        if kwargs.get('delete_folders', True):
            shutil.rmtree(simfolder_)

        return res

    #print('Starting simulation(s)...')
    output = p_map(
        _runner, 
        tasks, 
        num_cpus=njobs, 
        ncols=100, 
        desc='Progress', 
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        colour="#305069",
        ascii='-◼'
    )

    # Remove failed simulations
    output_cleaned = []
    input_cleaned = []
    for idx, res in enumerate(output):
        if res is None:
            if isinstance(input, list) or isinstance(input, np.ndarray):
                print(f'Warning: Simulation {idx} failed and will be removed from output.')
        else:
            output_cleaned.append(res)
            try:
                input_cleaned.append(input[idx])
            except:
                input_cleaned = input

    if isinstance(input, np.ndarray):
        input_cleaned = np.array(input_cleaned)
  
    

    # Check if output format is provided
    output_type = kwargs.get('output_type', output_type)

    # Convert to right output format
    if output_type == 'array':
        for n in range(len(output_cleaned)):
            output_cleaned[n] = dataframe_to_vec(output_cleaned[n])
        output_cleaned = np.squeeze(np.array(output_cleaned))

    elif output_type == 'dataframe' and len(output_cleaned) == 1:
        output_cleaned = output_cleaned[0]

    if kwargs.get('return_input', False):
        return input_cleaned, output_cleaned
    else:
        return output_cleaned


def get_sim_results(casename, datatypes=None, dates=None):
    try:
        case = Summary(casename)
        if dates == None:
            dates = case.report_dates
        res = case.pandas_frame(time_index=dates)

        if datatypes == None:
            return res
        else:
            return res[datatypes]
    except:
        return None
    

def simulate_mako(filename, simfolder='SIM', verbosity=None, **kwargs):
    # Render mako file
    filename = filename.split("/")[-1]
    output = os.path.join(simfolder, f'{filename}.DATA')
    os.makedirs(simfolder, exist_ok=True)

    # Use absolute path for template to avoid path issues in multiprocessing
    template_path = os.path.abspath(f'{filename}.mako')
    tmpl = Template(filename=template_path)
    with open(output, 'w') as f:
        f.write(tmpl.render(**kwargs))

    # Simulate without changing working directory (safer for multiprocessing)
    import subprocess
    
    if verbosity is None:
        result = subprocess.run(['flow', f'{filename}.DATA'], 
                              cwd=simfolder, 
                              capture_output=True, 
                              text=True)
    else:
        result = subprocess.run(['flow', f'{filename}.DATA'], 
                              cwd=simfolder)
    
    return result.returncode
    
def dataframe_to_vec(df: pd.DataFrame, datatypes: list = None):
    vec = np.array([])
    for col in df.columns:
            col_vec = df[col].to_numpy()
            vec = np.append(vec, col_vec)
    return np.array(vec)

def vec_to_dataframe(vec, datatypes, index):
    df = {key: [] for key in datatypes}
    idx = 0
    for key in datatypes:
        df[key] = vec[idx:idx+len(index)]
        idx += len(index)
    
    return pd.DataFrame(df, index=index)


def get_latest_restart_index(casename, folder, delete_files=True):
    os.chdir(folder)
    largest_number = -1
    largest_file = None

    for filename in os.listdir('.'):
        if filename.startswith(f'{casename}.X'):
            number = int(filename.split('X')[-1])
            if number > largest_number:
                largest_number = number
                largest_file = filename

    # Move file
    if largest_file:
        os.rename(largest_file, os.path.join('../restarts', largest_file))

    # Delete files
    if delete_files:
        for filename in os.listdir('.'):
            os.remove(filename)

    os.chdir('../')
    return largest_number

def delete_restart_files(casename, folder, min=82):
    # Delete all restart files with index higher than 'min'
    os.chdir(folder)
    for filename in os.listdir('.'):
        if filename.startswith(f'{casename}.X'):
            number = int(filename.split('X')[-1])
            if number > min:
                os.remove(filename)
    os.chdir('../')



def facility_consumption(results: pd.DataFrame, baseload=8, windpower=None):

    # Check for FOPR, FGPR, FWPR, FWIR in results
    if 'FOPR' not in results.columns:
        # Sum all columns with WOPR in their name
        fopr_cols = [col for col in results.columns if 'WOPR' in col]
        results['FOPR'] = results[fopr_cols].sum(axis=1)

    if 'FGPR' not in results.columns:
        # Sum all columns with WGPR in their name
        fgpr_cols = [col for col in results.columns if 'WGPR' in col]
        results['FGPR'] = results[fgpr_cols].sum(axis=1)
    
    if 'FWPR' not in results.columns:
        # Sum all columns with WWPR in their name
        fwpr_cols = [col for col in results.columns if 'WWPR' in col]
        results['FWPR'] = results[fwpr_cols].sum(axis=1)
    
    if 'FWIR' not in results.columns:
        # Sum all columns with WWIR in their name
        fwir_cols = [col for col in results.columns if 'WWIR' in col]
        results['FWIR'] = results[fwir_cols].sum(axis=1)


    # Power required by gas compressor
    Pcomp = power_demand_gas_compressor(results['FGPR'].values)

    # Power required by pump
    thp_max = np.maximum(results['WTHP:A5'].values, results['WTHP:A6'].values)
    pump_head = (thp_max-1)*10.199773339984054
    Ppump = power_demand_pump(results['FWIR'].values, Hreq=pump_head)

    # Power required by water treatment
    Pwt = power_demand_water_treatment(results['FWPR'].values)

    # Total power demand
    Ptotal = Pcomp + Ppump + Pwt + baseload  # [MW]

    # Subtract wind power if provided
    if windpower is not None:
        Pgas = np.maximum(Ptotal - windpower, 0)
    
    # Calculate fuel consumption from total power demand
    co2_rate, fuel_rate = turbine_system_consumption(Pgas) # [ton CO2/day], [ton fuel/day]

    # Subtract gas fule consumption from FGPR 0.7215 kg/Sm3
    density_gas = 0.7215 # kg/Sm3
    fuel_rate_Sm3_per_day = (fuel_rate * 1e3)/density_gas  # Sm3/day
    gas_export = np.maximum(results['FGPR'].values - fuel_rate_Sm3_per_day, 0)

    # Make dataframe
    df = {
        'emission_rate [ton/day]': co2_rate, 
        'fuel_rate [ton/day]': fuel_rate, 
        'power_demand [MW]': Ptotal,
        'gas_export [Sm3/day]': gas_export,
    }

    if windpower is not None:
        df['windpower [MW]'] = windpower

    df = pd.DataFrame(df, index=results.index)
    return df

def production_volume_and_emissions(results: pd.DataFrame=None, windpower=None, fac=None):
    # Calculate facility consumption
    if not results is None:
        fac = facility_consumption(results, windpower=windpower)
    elif fac is None:
        raise ValueError('Either results or fac must be provided')
    
    res = {
        'co2': np.sum(fac['emission_rate [ton/day]'].values),
        'oil': np.sum(results['FOPR'].values),
        'gas': np.sum(results['FGPR'].values),
        'gas_exp': np.sum(fac['gas_export [Sm3/day]'].values),
        'wi': np.sum(results['FWIR'].values),
        'wp': np.sum(results['FWPR'].values),
    }
    return res

def data_well(data: pd.DataFrame, wellname: str):
    # Extract data for a specific well from dataframe
    cols = [col for col in data.columns if wellname in col]
    return data[cols].copy()

def net_present_value(volumes: dict, econ: dict):
    revenue_oil = volumes['oil'] * econ['oil']
    revenue_gas = volumes['gas_exp'] * econ['gas']
    cost_wi     = volumes['wi']  * econ['wi']
    cost_wp     = volumes['wp']  * econ['wp']
    cost_co2    = volumes['co2'] * econ['co2']
    net_cash_flow = revenue_oil + revenue_gas - cost_wi - cost_wp - cost_co2
    return net_cash_flow


def print_diff(i, vo, vt, vp=None, filepath=None):
    
    # Delta true values
    dnpv = (vo['npv']/vt['npv']-1)*100
    dco2 = (vo['co2']/vt['co2']-1)*100
    doil = (vo['oil']/vt['oil']-1)*100
    dgas = (vo['gas']/vt['gas']-1)*100
    dwi  = (vo['wi']/vt['wi']-1)*100
    dwp  = (vo['wp']/vt['wp']-1)*100

    # Delta proxy values if provided
    if vp is not None:
        dnpv_p = (vp['npv']/vt['npv']-1)*100
        dco2_p = (vp['co2']/vt['co2']-1)*100
        doil_p = (vp['oil']/vt['oil']-1)*100
        dgas_p = (vp['gas']/vt['gas']-1)*100
        dwi_p  = (vp['wi']/vt['wi']-1)*100
        dwp_p  = (vp['wp']/vt['wp']-1)*100
    else:
        dnpv_p = dco2_p = doil_p = dgas_p = dwi_p = dwp_p = 0.00


    # Print Information
    print("\n" + "="*100)
    print("RESULTS FOR INTERVAL", i+1)
    print("="*100)
    print(f"{'Metric':<22} {'Optimized':>12} {'Target':>12} {'Unit':>12} {'Δ (%)':>12} {'Proxy Δ (%)':>18}")
    print("-"*100)
    print(f"{'Net Present Value':<22} {vo['npv']:>12.2f} {vt['npv']:>12.2f} {'$':>12} {dnpv:>12.2f} {dnpv_p:>18.2f}")
    print(f"{'CO2 Emissions':<22} {vo['co2']:>12.2f} {vt['co2']:>12.2f} {'ton':>12} {dco2:>12.2f} {dco2_p:>18.2f}")
    print(f"{'Oil Production':<22} {vo['oil']:>12.2f} {vt['oil']:>12.2f} {'Sm3':>12} {doil:>12.2f} {doil_p:>18.2f}")
    print(f"{'Gas Production':<22} {vo['gas']:>12.2f} {vt['gas']:>12.2f} {'Sm3':>12} {dgas:>12.2f} {dgas_p:>18.2f}")
    print(f"{'Water Injection':<22} {vo['wi']:>12.2f} {vt['wi']:>12.2f} {'Sm3':>12} {dwi:>12.2f} {dwi_p:>18.2f}")
    print(f"{'Water Production':<22} {vo['wp']:>12.2f} {vt['wp']:>12.2f} {'Sm3':>12} {dwp:>12.2f} {dwp_p:>18.2f}")
    print("="*100 + "\n")   

    # Write to file if filepath provided
    if filepath is not None:
        with open(filepath, 'a') as f:
            f.write("="*100 + "\n")
            f.write(f"RESULTS FOR INTERVAL {i+1}\n")
            f.write("="*100 + "\n")
            f.write(f"{'Metric':<22} {'Optimized':>12} {'Target':>12} {'Unit':>12} {'Δ':>12} {'Δ (proxy)':>18}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Net Present Value':<22} {vo['npv']:>12.2f} {vt['npv']:>12.2f} {'$':>12} {dnpv:>12.2f} {dnpv_p:>18.2f}\n")
            f.write(f"{'CO2 Emissions':<22} {vo['co2']:>12.2f} {vt['co2']:>12.2f} {'ton':>12} {dco2:>12.2f} {dco2_p:>18.2f}\n")
            f.write(f"{'Oil Production':<22} {vo['oil']:>12.2f} {vt['oil']:>12.2f} {'Sm3':>12} {doil:>12.2f} {doil_p:>18.2f}\n")
            f.write(f"{'Gas Production':<22} {vo['gas']:>12.2f} {vt['gas']:>12.2f} {'Sm3':>12} {dgas:>12.2f} {dgas_p:>18.2f}\n")
            f.write(f"{'Water Injection':<22} {vo['wi']:>12.2f} {vt['wi']:>12.2f} {'Sm3':>12} {dwi:>12.2f} {dwi_p:>18.2f}\n")
            f.write(f"{'Water Production':<22} {vo['wp']:>12.2f} {vt['wp']:>12.2f} {'Sm3':>12} {dwp:>12.2f} {dwp_p:>18.2f}\n")
            f.write("="*100 + "\n\n\n")