# Imports
import numpy as np
import pandas as pd
import datetime as dt

import sys
import os
import yaml

# Internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim_tools
import opt_tools


# Configuration
################################################################################
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Workflow parameters
startdate = config['workflow']['startdate']
enddate   = config['workflow']['enddate']
ddays     = config['workflow']['interval_length']
rst_index = config['workflow']['restart_index']

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

datatypes = []
for key, wells in config['simulation']['datatypes'].items():
    for well in wells:
        datatypes.append(f'{key}:{well}')
################################################################################


# Run loop
if __name__ == '__main__':

    # Initialize 
    current_date  = startdate
    current_month = dt.datetime.strptime(current_date, '%Y-%m-%d').month
    current_year  = dt.datetime.strptime(current_date, '%Y-%m-%d').year
    month_index   = 0
    day_index     = 0
    year_index    = 0

    # Number of intervals
    nint = 1 + (pd.to_datetime(enddate) - pd.to_datetime(startdate)).days // ddays
    print(f'Number of intervals: {nint}')

    # Load wind power file
    windpower = np.load('windpower.npy')

    # Loop over intervals
    for i in range(nint):
        '''
        For each interval:
            - Xi: Target controls
            - Yi: Simulated data from Xi
            - WPi: Wind power
            - Fi: Facility consumption
            - Vi: Production volumes (Sm3) and emission (ton)
        
        NB: _df indicates pandas DataFrame format, else numpy array format.
        '''
        print(f'\n=== Interval {i+1} / {nint} ===')


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
        
        Xi_df = {}
        for key in controls:
            Xi_df[key] = np.repeat(np.array(target.iloc[index][key]), ddays)
        Xi_df = pd.DataFrame(Xi_df, index=dates)
        Xi = sim_tools.dataframe_to_vec(Xi_df)


        # Load wind power for interval i
        WPi = windpower[day_index:day_index+ddays]


        # Simulate target strategy
        Yi_df = sim_tools.simulate(
            filename=filename,
            input=Xi_df,
            dates=dates,
            rst_idx=rst_index,
            output_keys=datatypes,
            njobs=1,
        )
        

        # calculate facility consumption
        Fi_df = sim_tools.facility_consumption(Yi_df, windpower=WPi)
        Vi_dict = sim_tools.production_volume_and_emissions(Yi_df, windpower=WPi)


        # Sample 100 random control strategies around target to train proxy
        size = 100
        cov = 0.25**2 * np.diag((ub - lb)**2)
        X = np.random.multivariate_normal(mean=Xi, cov=cov, size=size)
        X = np.clip(X, a_min=lb, a_max=ub)

        # Simulate sampled strategies
        if True:
            Y = sim_tools.simulate(
                filename=filename,
                input=X,
                input_keys=controls,
                dates=dates,
                rst_idx=rst_index,
                output_keys=datatypes,
                njobs=parallel,
            )
            # Save X and Y
            np.savez('proxy_data.npz', X=X, Y=Y)
        else:
            # Load X and Y 
            data = np.load('proxy_data.npz')
            X = data['X']
            Y = data['Y']


        # Make proxy model
        proxy_model = opt_tools.LinearProxyModel(X, Y)


        # Run target strategy on proxy model
        proxy_Yi = proxy_model.predict(Xi)
        proxy_Yi_df = sim_tools.vec_to_dataframe(
            vec=proxy_Yi.squeeze(), 
            datatypes=datatypes,
            index=dates
        )
        proxy_Fi_df = sim_tools.facility_consumption(proxy_Yi_df, windpower=WPi)
        proxy_Vi_dict = sim_tools.production_volume_and_emissions(proxy_Yi_df, windpower=WPi)


        # Define objective function
        objective = opt_tools.ObjectiveFunction(
            model=proxy_model, 
            target=proxy_Vi_dict, 
            windpower=WPi,
            weight=0.5,
        )


        # Minimize objective function
        if False:
            cov = 0.1**2 * np.diag((ub - lb)**2)
            U = np.random.multivariate_normal(mean=Xi, cov=cov, size=10000)
            U = np.clip(U, a_min=lb, a_max=ub)

            F = []
            for u in U:
                fval = objective(u, dates, datatypes)
                F.append(fval)
            F  = np.array(F)
            Ui = U[np.argmin(F)].squeeze()
        else:
            Ui = Xi/ub
            Ui = np.random.multivariate_normal(mean=Ui, cov=0.05**2 * np.eye(len(Ui)))
            func = lambda u, *args: objective(u*ub, *args)

            from scipy.optimize import minimize
            res = minimize(
                fun=func,
                x0=Ui,
                args=(dates, datatypes),
                method='L-BFGS-B',
                jac='3-point',
                bounds=[(0, 1) for _ in range(len(Ui))],
                options={
                    'disp': True,
                    'maxiter': 100,
                }
            )
            print(res)
            Ui = res.x * ub


        # Evaluate optimal strategy on full simulator
        Xi_df_opt = sim_tools.vec_to_dataframe(
            vec=Ui, 
            datatypes=controls,
            index=dates
        )
        Yi_df_opt = sim_tools.simulate(
            filename=filename,
            input=Xi_df_opt,
            dates=dates,
            rst_idx=rst_index,
            output_keys=datatypes,
            njobs=1,
            get_restart=True,
        )
        Fi_df_opt = sim_tools.facility_consumption(Yi_df_opt, windpower=WPi)
        Vi_dict_opt = sim_tools.production_volume_and_emissions(Yi_df_opt, windpower=WPi)
        
        print('\n=== Optimal strategy evaluation ===')
        objective.target = Vi_dict
        objective.func(Yi_df_opt, eval=True)

        # Save results
        result_folder = f'results/Interval{i+1}'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        target_folder = os.path.join(result_folder, 'target')
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        optimal_folder = os.path.join(result_folder, 'optimal')
        if not os.path.exists(optimal_folder):
            os.makedirs(optimal_folder)

        # Save controls
        Xi_df.to_csv(os.path.join(target_folder, 'controls.csv'))
        Xi_df_opt.to_csv(os.path.join(optimal_folder, 'controls.csv'))

        # Save simulation results
        Yi_df.to_csv(os.path.join(target_folder, 'simulation.csv'))
        Yi_df_opt.to_csv(os.path.join(optimal_folder, 'simulation.csv'))

        # Save facility consumption
        Fi_df.to_csv(os.path.join(target_folder, 'consumption.csv'))
        Fi_df_opt.to_csv(os.path.join(optimal_folder, 'consumption.csv'))

        # Calculate and save NPV in npz files
        npv_target = sim_tools.net_present_value(
            volumes=Vi_dict,
            econ=econ,
        )
        npv_optimal = sim_tools.net_present_value(
            volumes=Vi_dict_opt,
            econ=econ,
        )
        np.savez(os.path.join(target_folder, 'npv.npz'), npv=npv_target)
        np.savez(os.path.join(optimal_folder, 'npv.npz'), npv=npv_optimal)



        # Update for next interval
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
        

        


       
