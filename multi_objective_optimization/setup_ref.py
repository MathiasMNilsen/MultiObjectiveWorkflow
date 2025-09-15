import shutil
import numpy as np
import pandas as pd
import datetime as dt

from glob import glob

# Imports from PET
from popt.loop.ensemble_gaussian import GaussianEnsemble
from simulator.opm import flow
from input_output import read_config

# Internal import
from function import objectives

# Set random seed
np.random.seed(29_11_1997)

def dummy_objective(pred_data, input_dict, true_order):
    npv, co2 = objectives(pred_data, input_dict, true_order)
    np.savez('init/ref_values', pred_data=pred_data, co2=co2, npv=npv)
    return None

if __name__ == '__main__':
    
    # Configurate
    kwopt, kwsim, kwen = read_config.read_yaml('config.yaml')

    # Fix first reportpoint
    kwsim['reportpoint'][0] = dt.datetime(2020, 7, 2, 0, 0) 

    # Define dummy ensemble
    dummyens = GaussianEnsemble(kwen, flow(kwsim), objective=dummy_objective)
    dummyens.function(dummyens.get_state())