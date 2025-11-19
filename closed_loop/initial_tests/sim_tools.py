import numpy  as np
import pandas as pd
import os

from misc import ecl
from glob import glob
from mako.template import Template
from resdata.summary import Summary
from p_tqdm import p_imap

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
        return simulate_mako(fn, simfolder_, **df_d, rep_dates=dates, rst_index=rst_idx)

    # p_imap starts processing immediately (faster startup than p_map)
    print('\nStarting simulation(s)...')
    list(p_imap(_runner, tasks, num_cpus=njobs, ncols=100, desc='Progress'))
    print('\n')

    # Collect results
    output = []
    for n in range(ne):
        simfolder = f'Sim{n}'
        res = get_sim_results(
            casename=f'{simfolder}/{filename}',     
            datatypes=output_keys, 
            dates=dates
        )
        output.append(res)

    # Check if output format is provided
    output_type = kwargs.get('output_type', output_type)

    # Convert to right output format
    if output_type == 'array':
        for n in range(ne):
            output[n] = dataframe_to_vec(output[n])
        output = np.squeeze(np.array(output))

    elif output_type == 'dataframe' and len(output) == 1:
        output = output[0]

    # Get restart file
    if kwargs.get('get_restart', False):
        get_latest_restart_index(casename=filename, folder=simfolder)

    # Delete simulation folders
    if kwargs.get('delete_folders', True):
        for n in range(ne):
            simfolder = f'Sim{n}'
            import shutil
            shutil.rmtree(simfolder)

    return output


def get_sim_results(casename, datatypes=None, dates=None):
    case = Summary(casename)
    if dates == None:
        dates = case.report_dates
    res = case.pandas_frame(time_index=dates)

    if datatypes == None:
        return res
    else:
        return res[datatypes]
    

def simulate_mako(filename, simfolder='SIM', verbosity=None, **kwargs):
    # Render mako file
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
    
    if result.returncode != 0:
        raise RuntimeError(f'Flow simulation failed in {simfolder} with return code {result.returncode}')

    return None
    
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



def facility_consumption(results: pd.DataFrame, baseload=4, windpower=None):

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
        'gas': np.sum(fac['gas_export [Sm3/day]'].values),
        'wi': np.sum(results['FWIR'].values),
        'wp': np.sum(results['FWPR'].values),
    }
    return res

def net_present_value(volumes: dict, econ: dict):
    revenue_oil = volumes['oil'] * econ['oil']
    revenue_gas = volumes['gas'] * econ['gas']
    cost_wi     = volumes['wi']  * econ['wi']
    cost_wp     = volumes['wp']  * econ['wp']
    cost_co2    = volumes['co2'] * econ['co2']
    net_cash_flow = revenue_oil + revenue_gas - cost_wi - cost_wp - cost_co2
    return net_cash_flow