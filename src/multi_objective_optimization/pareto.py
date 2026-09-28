# Imports
import numpy as np
import datetime as dt

# Imports from PET
from popt.loop.ensemble_gaussian import GaussianEnsemble
#from input_output import read_config
from subsurface.multphaseflow.opm import flow

# Import objective function
from multi_objective_optimization.function import objectives

# Function to optimize a single Pareto point
def optimize_pareto_point(weight, kwopt, kwsim, kwens, **kwargs):

    # Objective function
    def weighted_sum(pred_data, input_dict, true_order, save=False, **kwargs):
        # Calculate NPV and CO2
        npv, co2, capex = objectives(
            pred_data,
            input_dict=input_dict,
            true_order=true_order,
            **kwargs
        )
        # npv.shape -> (ne,)
        # co2.shape -> (ne, ndays)

        # Save data (if save = True)
        if save:
            np.savez(
                f'{kwopt['save_folder']}/pareto_point',
                pred_data=pred_data,
                co2=co2,
                npv=npv,
                capex=capex,
            )

        # Get objective scaling from kwargs (if applicable)
        obj_scaling = kwopt.get('obj_scaling', [1,1,1])
        if isinstance(obj_scaling, (float, int)):
            obj_scaling = [obj_scaling] * 3

        # Get f1 (either CO2 or CAPEX) and f2 (NPV)
        f1 = co2.sum(axis=1)
        # Use CAPEX if CO2 values are very small and CAPEX is available
        if np.all(f1 < 1.0e-6) and np.any(capex):
            f1 = capex
        f2 = npv

        # Calculate  weighted sum
        wsum = weight * f1 / obj_scaling[0] + (1 - weight) * (-f2 / obj_scaling[1])

        # Get the penalty from kwargs (if applicable) and add to weighted sum
        epf_dict = kwargs.get('epf', {})
        if epf_dict:
            epf_dict['penalty'] = np.sum(epf_dict['penalty'], axis=0)
            epf_dict['penalty'] /= obj_scaling[2]
            wsum += epf_dict['penalty']

        return wsum


    # Define ensemble
    ensemble = GaussianEnsemble(kwens, flow(kwsim), objective=weighted_sum)

    # Initial state
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    # Read optimal control strategy from a previous optimization (for analysis purposes)
    #f = np.load('../results/run2w0.5/optimize_result_3_4.npz', allow_pickle=True)
    #x0 = f['x']

    epf = None
    main_method = kwopt.get('main_method', 'LineSearch')
    if main_method == 'LineSearch':
        from popt.update_schemes.linesearch import LineSearch
        # Define callables
        normalize = kwopt.get('normalize', False)
        if normalize:
            scaling = cov[0, 0]
        else:
            scaling = 1.0
        func = lambda x, *args, **kwargs: ensemble.function(x, *args, **kwargs)
        grad = lambda x, *args, **kwargs: ensemble.gradient(x, *args, **kwargs) / scaling
        hess = lambda x, *args: np.diag(np.diag(ensemble.hessian(x, *args))) / scaling ** 2
        # Run optimization
        res = LineSearch(fun=func,x=x0,jac=grad,hess=hess,args=(cov,),bounds=bounds,**kwopt)
        x = res.x
        if hasattr(res, 'epf'):
            epf = res.epf
    elif main_method == 'EnOpt':
        from popt.update_schemes.enopt import EnOpt
        opt_obj = EnOpt(ensemble.function, x0, args=(cov,), jac=ensemble.gradient,
                        hess=ensemble.hessian, bounds=bounds, **kwopt)
        x = opt_obj.xk
        epf = opt_obj.epf
    else:
        raise ValueError(f'Unknown main_method: {main_method}')

    # Get final CO2 and NPV
    def dummy_func(pred_data, input_dict, true_order, **kwargs):
        return weighted_sum(pred_data, input_dict, true_order, save=True, **kwargs)

    ensemble.obj_func = dummy_func
    ensemble.function(x, epf=epf)
