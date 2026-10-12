# Imports
import os
from copy import deepcopy

import numpy as np
import pandas as pd

# Imports from PET
from popt import GaussianEnsemble, LineSearch, EnOpt
from subsurface.multphaseflow.opm import flow

# Import objective function
from multi_objective_optimization.function import objectives

# Options read by this module only; everything else in kwopt goes to the PET optimizer
_PARETO_OPTIONS = ('main_method', 'obj_scaling')


# Function to optimize a single Pareto point
def optimize_pareto_point(weight, kwopt, kwsim, kwens, **kwargs):

    # Objective function
    def weighted_sum(pred_data, input_dict, true_order, save=False, **kwargs):
        # PET hands over the forecast as a frame; objectives() works on one dict per report step
        pred_data = to_pred_data(pred_data)

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
                f'{kwopt["save_folder"]}/pareto_point',
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

    # Initial state, covariance and bounds, in the original units of the controls
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    # Read optimal control strategy from a previous optimization (for analysis purposes)
    #f = np.load('../results/run2w0.5/optimize_result_3_4.npz', allow_pickle=True)
    #x0 = f['x']

    # With kwens['transform'], the optimizer works on the controls scaled to the unit cube, u = (x - lb)/(ub - lb),
    # with the covariance scaled to match. Earlier PET versions did this inside the ensemble; the current one
    # works in the original units, and its own `transform` option would also rescale EnOpt's covariance update
    # differently. Doing it here gives the optimizer the same gradient and Hessian as before: the natural
    # (default) ensemble gradient in x is D times the one in u, and the Hessian D H D, with D = diag(ub - lb).
    if kwens.get('transform', False):
        lb, ub = np.asarray(bounds, dtype=float).T
        bounds = [(0.0, 1.0)] * x0.size
    else:
        lb, ub = np.zeros(x0.size), np.ones(x0.size)
    db = ub - lb
    to_x = lambda u: lb + np.asarray(u) * db
    x0 = (x0 - lb) / db
    cov = cov / np.outer(db, db)

    # Optional scaling of the gradient by the (unit-cube) variance, for LineSearch only (EnOpt normalizes itself)
    scaling = cov[0, 0] if kwopt.get('normalize', False) and kwopt.get('main_method', 'LineSearch') == 'LineSearch' else 1.0

    func = lambda u, *args, **kwargs: ensemble.function(to_x(u), **kwargs)
    grad = lambda u, cov, **kwargs: ensemble.gradient(to_x(u), cov * np.outer(db, db), **kwargs) / db / scaling
    hess = lambda u, cov, **kwargs: (ensemble.hessian(to_x(u), cov * np.outer(db, db), **kwargs)
                                     / np.outer(db, db) / scaling ** 2)

    # Optimizer options. A copy, since PET updates the EPF dict (penalty factor) in place and kwopt is reused
    # for every weight. PET's own transform stays off (see above), and the iterations are saved by
    # save_iteration below, in the layout plot_results reads.
    options = {key: value for key, value in deepcopy(kwopt).items() if key not in _PARETO_OPTIONS}
    options['transform'] = False
    options['saveit'] = False
    save_folder = kwopt.get('save_folder', '.')
    callback = lambda opt: save_iteration(opt, save_folder)

    main_method = kwopt.get('main_method', 'LineSearch')
    if main_method == 'LineSearch':
        # Newton-CG uses only the diagonal of the ensemble Hessian
        diag_hess = lambda u, cov, **kwargs: np.diag(np.diag(hess(u, cov, **kwargs)))
        optimizer = LineSearch(x0, func, jac=grad, hess=diag_hess, args=(cov,), bounds=bounds,
                               callback=callback, **options)
    elif main_method == 'EnOpt':
        optimizer = EnOpt(x0, func, jac=grad, hess=hess, args=(cov,), bounds=bounds, callback=callback, **options)
    else:
        raise ValueError(f'Unknown main_method: {main_method}')

    # Run optimization
    optimizer.run_optimization()
    x = to_x(optimizer.optimize_results['x'])
    epf = optimizer.epf

    # Get final CO2 and NPV
    def dummy_func(pred_data, input_dict, true_order, **kwargs):
        return weighted_sum(pred_data, input_dict, true_order, save=True, **kwargs)

    ensemble.obj_func = dummy_func
    ensemble.function(x, epf=epf)


def to_pred_data(sim_data):
    """Convert PET's forecast frame (one row per report date, one array over the members per cell) to a list
    with one dict per report date, mapping each data type to an array of shape (1, ne). Lists pass through."""
    if not isinstance(sim_data, pd.DataFrame):
        return sim_data
    return [{key: np.atleast_1d(np.asarray(value, dtype=float))[np.newaxis, :] for key, value in row.items()}
            for _, row in sim_data.iterrows()]


def save_iteration(optimizer, folder):
    """Save an accepted iteration as optimize_result_<it>.npz, or optimize_result_<epf_it>_<it>.npz with an EPF
    outer loop (counted from 0), the file layout plot_results reads.

    Used as the optimizer callback, which PET calls after every accepted step. The first step of each outer
    loop also saves the point it started from as iteration 0. x is in the optimizer's coordinates: the unit
    cube when kwens['transform'] is set.
    """
    def save(iteration, x, fun):
        if optimizer.epf:
            name = f'optimize_result_{optimizer.epf_iteration - 1}_{iteration}'
            np.savez(os.path.join(folder, name), x=x, fun=fun, epf=dict(optimizer.epf))
        else:
            np.savez(os.path.join(folder, f'optimize_result_{iteration}'), x=x, fun=fun)

    if optimizer.iteration == 1:
        save(0, optimizer.xk_old, optimizer.fk_old)
    save(optimizer.iteration, optimizer.xk, optimizer.fk)
