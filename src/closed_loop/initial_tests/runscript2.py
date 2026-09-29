# Imports
import numpy as np
import pandas as pd
import datetime as dt

import sys
import os
import yaml

from multiprocessing import Pool
from tqdm import tqdm
from scipy.optimize import minimize

# Internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim_tools
import opt_tools


# Configuration
################################################################################
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Workflow parameters
startdate   = config['workflow']['startdate']
enddate     = config['workflow']['enddate']
ddays       = config['workflow']['interval_length']
rst_index   = config['workflow']['restart_index']
save_folder = config['workflow']['save_folder']

# Load Target
target_file = config['workflow']['target_file']
target = pd.read_csv(target_file, index_col=0, parse_dates=True)

# Economic parameters
econ = {
    'oil': config['economic']['oil_price'],
    'gas': config['economic']['gas_price'],
    'wi' : config['economic']['water_injection'],
    'wp' : config['economic']['water_production'],
    'co2': config['economic']['co2_tax']
}

# Simulation parameters
filename = config['simulation']['filename']
parallel = config['simulation']['parallel']
controls = [key for key in config['simulation']['controls'].keys()]

lb = np.array([])
ub = np.array([])
for key in controls:
    lb = np.append(lb, [config['simulation']['controls'][key]['min']]*ddays)
    ub = np.append(ub, [config['simulation']['controls'][key]['max']]*ddays)

#datatypes = []
#for key, wells in config['simulation']['datatypes'].items():
#    for well in wells:
#        datatypes.append(f'{key}:{well}')

datatypes = ['FOPR', 'FGPR', 'FWPR', 'FWIR','WTHP:A5', 'WTHP:A6']
################################################################################

# Delete old restart files
sim_tools.delete_restart_files(filename, 'restarts', min=rst_index)

# Delete previous results folder if exists
if os.path.exists(save_folder):
    import shutil
    shutil.rmtree(save_folder)

# Run loop
if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1_11_2018) # Lucky's birthday

    # Initialize 
    current_date  = startdate
    current_month = dt.datetime.strptime(current_date, '%Y-%m-%d').month
    current_year  = dt.datetime.strptime(current_date, '%Y-%m-%d').year
    month_index   = 0
    day_index     = 0
    year_index    = 0

    # Number of intervals
    nint = 1 + (pd.to_datetime(enddate) - pd.to_datetime(startdate)).days // ddays
    #nint = 18 
    print(f'Number of intervals: {nint}')

    # Create save folder and initialize results summary file
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Initialize results summary file (clear if exists)
    with open(f'{save_folder}/results_summary.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("OPTIMIZATION RESULTS SUMMARY\n")
        f.write("="*100 + "\n\n")

    # Load wind power file
    windpower = np.load('windpower.npy')

    # Loop over intervals
    for i in range(nint):
        '''
        For each interval:
            - xi_vec: Target controls (numpy array)
            - xi_df: Target controls (pandas DataFrame)
            - yi_vec: Simulated data from target controls (numpy array)
            - yi_df: Simulated data from target controls (pandas DataFrame)
            - wpi_vec: Wind power (numpy array)
            - fi_df: Facility consumption (pandas DataFrame)
            - vi_dict: Production volumes (Sm3) and emissions (ton) (dict)
        
        NB: _df suffix indicates pandas DataFrame format
            _dict suffix indicates dictionary format
            otherwise numpy array format
        '''
        s = f'Interval {i+1} / {nint}'
        l = (98 - len(s))//2
        print('\n' + '='*l + ' ' + s + ' ' + '='*l)


        # Reporting dates
        dates = pd.date_range(
            current_date, 
            periods=ddays, 
            freq='D'
        ).to_pydatetime().tolist()


        # Load target for interval i
        index = 0
        for date in target.index:
            if (date.month == current_month) and (date.year == current_year):
                break
            index += 1
        
        xi_df = {}
        for key in controls:
            xi_df[key] = np.repeat(np.array(target.iloc[index][key]), ddays)
        xi_df = pd.DataFrame(xi_df, index=dates)
        xi_vec = sim_tools.dataframe_to_vec(xi_df)

        # Load wind power for interval i
        wpi_vec = windpower[day_index:day_index+ddays]

        print('\nSimulating target strategy...')
        print("-" * 100)
        # Simulate target strategy
        yi_df = sim_tools.simulate(
            filename=filename,
            input=xi_df,
            dates=dates,
            rst_idx=rst_index,
            output_keys=datatypes,
            njobs=1,
        )
        print("-" * 100)

        # calculate facility consumption
        fi_df = sim_tools.facility_consumption(yi_df, windpower=wpi_vec)
        vi_dict = sim_tools.production_volume_and_emissions(yi_df, windpower=wpi_vec)

        # Calculate NPV for target strategy
        npv_target = sim_tools.net_present_value(
            volumes=vi_dict,
            econ=econ,
        )
        vi_dict['npv'] = npv_target

        
        # MAKE PROXY MODEL
        ####################################################################################################
        if True:
            size = 1000
            cov  = 0.1**2 * np.diag((ub - lb)**2)
            X = np.random.multivariate_normal(mean=xi_vec, cov=cov, size=size)
            X = np.clip(X, a_min=lb, a_max=ub)

            print('\nSimulating sampled strategies for training proxy model...')
            print("-" * 100)
            # Simulate sampled strategies
            Y = sim_tools.simulate(
                filename=filename,
                input=X,
                input_keys=controls,
                dates=dates,
                rst_idx=rst_index,
                output_keys=datatypes,
                njobs=parallel,
            )
            print("-" * 100)

            # Train proxy model
            #proxy_model = opt_tools.LinearProxyModel(X, Y)
            #proxy_model = opt_tools.PolyProxyModel(X, Y, degree=2)
            proxy_model = opt_tools.RBFProxyModel(X, Y)


            # Run target strategy on proxy
            yi_vec_proxy  = proxy_model.predict(xi_vec)
            yi_df_proxy   = sim_tools.vec_to_dataframe(yi_vec_proxy, datatypes, index=dates)
            fi_df_proxy   = sim_tools.facility_consumption(yi_df_proxy, windpower=wpi_vec)
            vi_dict_proxy = sim_tools.production_volume_and_emissions(yi_df_proxy, windpower=wpi_vec)

            # Calculate NPV for target strategy on proxy
            npv_target_proxy = sim_tools.net_present_value(
                volumes=vi_dict_proxy,
                econ=econ,
            )
            vi_dict_proxy['npv'] = npv_target_proxy
        ####################################################################################################



        # SHORT-TERM OPTIMIZATION (CONSTRAINED FORMULATION)
        ####################################################################################################
        if True:
            # Transform target controls to [0, 1] space
            u0 = (xi_vec - lb) / (ub - lb)
            bounds = [(0, 1)] * u0.size

            # Define relative difference function setter
            def rel_diff(key):
                def _func(u):
                    x = u * (ub - lb) + lb
                    y = proxy_model.predict(x)
                    y = sim_tools.vec_to_dataframe(y, datatypes, index=dates)
                    vol = sim_tools.production_volume_and_emissions(y, windpower=wpi_vec)[key]
                    vol_target = vi_dict_proxy[key]
                    #vol_target = vi_dict[key]
                    return (vol - vol_target) / vol_target
                return _func
            
            def rel_diff_npv(u):
                x = u * (ub - lb) + lb
                y = proxy_model.predict(x)
                y = sim_tools.vec_to_dataframe(y, datatypes, index=dates)
                vol = sim_tools.production_volume_and_emissions(y, windpower=wpi_vec)
                npv = sim_tools.net_present_value(volumes=vol, econ=econ)
                npv_target = vi_dict_proxy['npv']
                return (npv - npv_target) / abs(npv_target)
            
            # Vectorized objective function for CO2 emissions
            #co2 = np.vectorize(rel_diff('co2'), signature='(n)->()')
            co2 = rel_diff('co2')

            # Define tolerance and constraints
            tol = 0.001  # 1% tolerance

            doil = rel_diff('oil')
            dgas = rel_diff('gas')
            dwi  = rel_diff('wi')
            dwp  = rel_diff('wp')

            def constraint_dco2_dnpv(u):
                return abs(co2(u)) - abs(rel_diff_npv(u))*2
            
            constraints = [
                {'type': 'ineq', 'fun': lambda u: tol - abs(doil(u))},
                {'type': 'ineq', 'fun': lambda u: tol - abs(dgas(u))},
                {'type': 'ineq', 'fun': lambda u: tol - abs(dwi(u))},
                {'type': 'ineq', 'fun': lambda u: tol - abs(dwp(u))},
                #{'type': 'ineq', 'fun': constraint_dco2_dnpv},
            ]

            print('\nRunning short-term optimization...')
            print("=" * 100)

            # Callback to monitor optimization with professional progress tracking
            maxiter = 5000
            progbar = tqdm(
                total=maxiter,
                desc='Short-term Optimization',
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                colour="#305069"
            )
            it = [0]  # Use list to allow modification in nested function
            
            def callback(xk):
                progbar.update(1)
                progbar.set_description(f'Func: {co2(xk):.6f}')
                it[0] += 1

            # Minimize CO2 emissions
            res = minimize(
                fun=co2,
                x0=u0,
                method='COBYLA',
                bounds=bounds,
                constraints=constraints,
                callback=callback,
                options={'disp': False, 'maxiter': maxiter, 'rhobeg': 0.25},
            )
            progbar.close()

            if res.success:
                print(res)
                xi_vec_opt = res.x * (ub - lb) + lb
            else:
                print("Optimization failed, using target controls as fallback.")
                xi_vec_opt = xi_vec  # Fallback to target controls

            print("=" * 100)
        ####################################################################################################



        # SHORT-TERM OPTIMIZATION (WEIGHTED FORMULATION)
        ####################################################################################################
        if False:
            u0 = (xi_vec - lb) / (ub - lb)
            bounds = [(0, 1)] * u0.size

            # Sample 50000 random points and pick the best as initial guess
            size = 1000
            cov  = 0.1**2 * np.eye(u0.size)
            U = np.random.multivariate_normal(mean=u0, cov=cov, size=size)
            U = np.clip(U, a_min=0, a_max=1)
            
            # Objective function
            w = 0.1
            def objective(u):
                x = u * (ub - lb) + lb
                y = proxy_model.predict(x)
                y = sim_tools.vec_to_dataframe(y, datatypes, index=dates)
                vol_target = vi_dict_proxy
                vol = sim_tools.production_volume_and_emissions(y, windpower=wpi_vec)
                dco2 = vol['co2']/vol_target['co2'] - 1
                doil = vol['oil']/vol_target['oil'] - 1
                dgas = vol['gas']/vol_target['gas'] - 1
                dwi  = vol['wi']/vol_target['wi'] - 1
                dwp  = vol['wp']/vol_target['wp'] - 1
                sqrt_sum = np.sqrt(doil**2 + dgas**2 + dwi**2 + dwp**2)
                fval = w*dco2 + (1 - w)*sqrt_sum
                return fval

            # Vecotrized objective function
            #objective = np.vectorize(objective, signature='(n)->()')

            print('\nRunning short-term optimization...')
            print("=" * 100)

            # Evaluate objective function on samples
            with Pool(processes=parallel) as pool:
                F = list(tqdm(
                    pool.imap(objective, U), 
                    total=len(U), 
                    desc='Short-term Optimization',
                    ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    colour='green'
                ))
            
            Ubest = U[np.argmin(F)].squeeze()
            xi_vec_opt = Ubest * (ub - lb) + lb
            print("=" * 100)
        ####################################################################################################



        # EVALUATE OPTIMAL STRATEGY
        ####################################################################################################
        # Simulate optimal strategy on proxy
        yi_vec_opt_proxy  = proxy_model.predict(xi_vec_opt)
        yi_df_opt_proxy   = sim_tools.vec_to_dataframe(yi_vec_opt_proxy, datatypes, index=dates)
        fi_df_opt_proxy   = sim_tools.facility_consumption(yi_df_opt_proxy, windpower=wpi_vec)
        vi_dict_opt_proxy = sim_tools.production_volume_and_emissions(yi_df_opt_proxy, windpower=wpi_vec)

        # Evaluate NPV for optimal strategy on proxy
        npv_opt_proxy = sim_tools.net_present_value(vi_dict_opt_proxy, econ=econ)
        vi_dict_opt_proxy['npv'] = npv_opt_proxy


        # Simulate optimal strategy on full model
        xi_df_opt = sim_tools.vec_to_dataframe(xi_vec_opt, controls, index=dates)
        yi_df_opt = sim_tools.simulate(
            filename=filename,
            input=xi_df_opt,
            dates=dates,
            rst_idx=rst_index,
            output_keys=datatypes,
            njobs=1,
            get_restart=True,
        )
        fi_df_opt = sim_tools.facility_consumption(yi_df_opt, windpower=wpi_vec)
        vi_dict_opt = sim_tools.production_volume_and_emissions(yi_df_opt, windpower=wpi_vec)

        # Evaluate NPV for optimal strategy on full model
        npv_opt = sim_tools.net_present_value(vi_dict_opt, econ=econ)
        vi_dict_opt['npv'] = npv_opt
        ####################################################################################################


        sim_tools.print_diff(
            i, 
            vi_dict_opt, 
            vi_dict, 
            vi_dict_opt_proxy, 
            filepath=f'{save_folder}/results_summary.txt'
        )
        

        # SAVE RESULTS
        ####################################################################################################
        result_folder = os.path.join(save_folder, f'Interval{i+1}')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        target_folder = os.path.join(result_folder, 'target')
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        optimal_folder = os.path.join(result_folder, 'optimal')
        if not os.path.exists(optimal_folder):
            os.makedirs(optimal_folder)

        proxy_folder = os.path.join(result_folder, 'proxy')
        if not os.path.exists(proxy_folder):
            os.makedirs(proxy_folder)

        # Save target results
        xi_df.to_csv(f'{target_folder}/controls.csv')
        yi_df.to_csv(f'{target_folder}/production.csv')
        fi_df.to_csv(f'{target_folder}/consumption.csv')
        np.savez(f'{target_folder}/volumes.npz', **vi_dict)

        # Save optimal results
        xi_df_opt.to_csv(f'{optimal_folder}/controls.csv')
        yi_df_opt.to_csv(f'{optimal_folder}/production.csv')
        fi_df_opt.to_csv(f'{optimal_folder}/consumption.csv')
        np.savez(f'{optimal_folder}/volumes.npz', **vi_dict_opt)

        # Save proxy results
        yi_df_opt_proxy.to_csv(f'{proxy_folder}/production.csv')
        fi_df_opt_proxy.to_csv(f'{proxy_folder}/consumption.csv')
        np.savez(f'{proxy_folder}/volumes.npz', **vi_dict_opt_proxy)
        ####################################################################################################



        # UPDATE FOR NEXT INTERVAL
        ####################################################################################################
        current_date = (pd.to_datetime(current_date) + dt.timedelta(days=ddays)).strftime('%Y-%m-%d')
        next_month = dt.datetime.strptime(current_date, '%Y-%m-%d').month
        next_year  = dt.datetime.strptime(current_date, '%Y-%m-%d').year

        if next_month != current_month:
            current_month = next_month
            month_index += 1

        if next_year != current_year:
            current_year = next_year
            year_index += 1
        
        day_index += ddays

        # Get restart index for next interval (find it in restarts folder)
        rst_index = sim_tools.get_latest_restart_index(
            casename=filename,
            folder=f'restarts',
            delete_files=False,
        )
        print(f'\nNext restart index: {rst_index}')
        ####################################################################################################
