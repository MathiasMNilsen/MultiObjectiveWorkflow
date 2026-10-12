"""Initial rates of the life-cycle optimization (data/closed_loop/init_rates.npz): nine six-month steps per well."""
import numpy as np


def write_initial_rates(path, n_steps=9):
    d = {
        'RATE_A1': np.array(n_steps*[2500]),
        'RATE_A2': np.array(n_steps*[2500]),
        'RATE_A3': np.array(n_steps*[2500]),
        'RATE_A4': np.array(n_steps*[2500]),
        'RATE_OP5': np.array(n_steps*[2500]),
        'RATE_A5': np.array(n_steps*[6000]),
        'RATE_A6': np.array(n_steps*[6000]),
    }
    np.savez(path, **d)
