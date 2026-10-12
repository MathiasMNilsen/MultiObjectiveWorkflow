"""Life-cycle optimization of the target rates (NPV) with PET.

    python examples/run-target_optimization.py

The controls are the rates of the wells, changed every six months (data/config/target_optimization.yaml,
data/closed_loop/DROGON_target.mako). The optimizer is the line search with Newton-CG set in the config. It saves
every iteration and the optimum (stateX.csv) to <run_dir>/<savefolder>. target_controls.py then turns the optimum
into the target of the closed loop.
"""
from pathlib import Path
import datetime as dt
import os
import shutil

# imports from PET
from popt import GaussianEnsemble, LineSearch
from input_output import read_config

# imports from SimulatorWrap
from subsurface.cost_functions import npv
from subsurface.multphaseflow.opm import flow

# Define objective function (NPV). A def, not a lambda: PET pickles the ensemble,
# objective included, when a simulation crashes.
def NPV(*args, **kwargs):
    return -npv(*args, **kwargs)/1e9


def optimize(config_file, deck, model_dir, run_dir):
    """Run the optimization in `run_dir` and return PET's result.

    PET runs the simulations in En_<member> folders of the working directory and reads the deck template from
    there, so the run happens in `run_dir`, with `deck` copied in as <runfile>.mako. The initial rates in the
    config are relative to the config file, and `model_dir` holds the include/ and Drogon_Mean/ data.
    """
    config_file, deck, model_dir, run_dir = (Path(p).resolve() for p in (config_file, deck, model_dir, run_dir))

    # Read config-file (the random seed is set in its ensemble section)
    kwopt, kwsim, kwens = read_config.read(str(config_file))
    for control in kwens['controls'].values():
        if isinstance(control.get('initial'), str):
            control['initial'] = str(config_file.parent / control['initial'])

    # Modify first report point
    kwsim['reportpoint'][0] = dt.datetime(2020, 7, 2, 0, 0)

    # The deck reads the model from model_dir, relative to the member folders En_<member>
    kwsim['mako_kwargs'] = {'model_dir': os.path.relpath(model_dir, run_dir / 'En_0')}

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(deck, run_dir / f"{kwsim['runfile']}.mako")
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        # Ensemble initialization
        ensemble = GaussianEnsemble(kwens, flow(kwsim), NPV)

        # Get initial state
        x0  = ensemble.get_state()
        cov = ensemble.get_cov()
        bounds = ensemble.get_bounds()

        # Run Line-search optimization (method, savefolder, ... from config-file)
        res = LineSearch.minimize(
            x0=x0,
            fun=ensemble.function,
            jac=ensemble.gradient,
            hess=ensemble.hessian,
            args=(cov,),
            bounds=bounds,
            **kwopt
        )
        print(res)

        # Save final state
        ensemble.save_stateX(state=res.x, path=f"{kwopt['savefolder']}/", filetype='csv')
    finally:
        os.chdir(cwd)
    return res
