import numpy as np
import datetime as dt

# imports from PET
from popt.loop.ensemble_gaussian import GaussianEnsemble
from popt.update_schemes.linesearch import LineSearch
from popt.update_schemes.trust_region import TrustRegion
from popt.cost_functions.npv import npv
from input_output import read_config
from subsurface.multphaseflow.opm import flow

# Define objective function (NPV)
NPV = lambda *args, **kwargs: -npv(*args, **kwargs)/1e9

if __name__ == '__main__':

    # Set random seed
    np.random.seed(29_11_1997)

    # Read config-file
    kwopt, kwsim, kwens = read_config.read('config.yaml')

    # Modify first report point
    kwsim['reportpoint'][0] = dt.datetime(2020, 7, 2, 0, 0) 
 
    # Ensemble initialization
    ensemble = GaussianEnsemble(kwens, flow(kwsim), NPV)

    # Get initial state
    x0  = ensemble.get_state()
    cov = ensemble.get_cov()
    bounds = ensemble.get_bounds()

    # Define function and gradient for optimization
    func = lambda x, *args, **kwargs: ensemble.function(x, *args, **kwargs)
    grad = lambda x, *args, **kwargs: ensemble.gradient(x, *args, **kwargs) 
    hess = lambda x, *args, **kwargs: ensemble.hessian(x, *args, **kwargs)


    '''
    res = LineSearch(
        fun=func,
        x=x0,
        jac=grad,
        hess=hess,
        args=(cov,),
        bounds=bounds,
        **kwopt
    )
    print(res)
    '''

    # Run Trust-region optimization
    res = TrustRegion(
        fun=func,
        x=x0,
        jac=grad,
        hess='BFGS',
        method='CG-Steihaug',
        args=(cov,),
        bounds=bounds,
        **{'save_folder': 'results_tr_BFGS_ncg', 'maxiter': 20}
    )
    print(res)

    # Save final state
    ensemble.save_stateX(path='results_tr_BFGS_ncg/', filetype='csv')