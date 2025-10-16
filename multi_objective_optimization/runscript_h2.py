# Imports 
import numpy as np
import datetime as dt

# Imports from PET
from popt.loop.ensemble_gaussian import GaussianEnsemble
#from popt.update_schemes.linesearch import LineSearch
from popt.update_schemes.enopt import EnOpt
from input_output import read_config
from simulator.opm import flow

# Import objective function
from function import *

# Load reference values (for scaling)
file = np.load('init/ref_values.npz', allow_pickle=True)
f1_ref = file['co2'].sum(axis=1).mean()
f2_ref = file['npv'].mean()

# Function to optimize a single Pareto point
def optimize_pareto_point(weight, save_folder):

    # Objective function
    def weighted_sum(pred_data, input_dict, true_order, save=False, **kwargs):

        # Calculate NPV and CO2
        npv, co2 = objectives(
            pred_data, 
            input_dict,
            true_order,
            **kwargs
        )
        # npv.shape -> (ne,)
        # co2.shape -> (ne, ndays)

        # Save data (if save = True)
        if save:
            np.savez(
                f'{save_folder}/pareto_point', 
                pred_data = pred_data,
                co2 = co2,
                npv = npv,
            )

        # Weighted sum
        f1 = co2.sum(axis=1)
        f2 = npv
        wsum = weight*f1/f1_ref + (1-weight)*(-f2/f2_ref)
        return wsum
    
    # Read config file
    kwopt , kwsim, kwen = read_config.read_yaml('config_h2.yaml')

    # Fix first reportpoint
    kwsim['reportpoint'][0] = dt.datetime(2020, 7, 2, 0, 0) 
     
    # Define ensemble
    ensemble = GaussianEnsemble(kwen, flow(kwsim), objective=weighted_sum)

    # Initial state
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    # Define callables
    func = lambda x,*args,**kwargs: ensemble.function(x,*args, **kwargs)
    grad = lambda x,*args,**kwargs: ensemble.gradient(x,*args, **kwargs)
    hess = lambda : ensemble.hessian()

    # Set options for line search
    options = {
        'save_folder': save_folder,
        'maxiter': 20,
        'step_size_adapt': 2,
        'ftol': 1e-5
    }

    # Run optimization
    res = None
    EnOpt(
        fun=func,
        x=x0,
        jac=grad,
        hess=hess,
        args=(cov,),
        bounds=bounds,
        **kwopt
    )

    # Get final CO2 and NPV
    def dummy_func(pred_data, input_dict, true_order):
        return weighted_sum(pred_data, input_dict, true_order, save=True)

    if res:
        ensemble.obj_func = dummy_func
        ensemble.function(res.x)
    
    

if __name__ == '__main__':

    # Set random seed and run
    np.random.seed(29_11_1997)
    optimize_pareto_point([0.0], save_folder=f'results/hydrogen')
