import numpy as np
d = {
    'RATE_A1': np.array(9*[2500]),
    'RATE_A2': np.array(9*[2500]),
    'RATE_A3': np.array(9*[2500]),
    'RATE_A4': np.array(9*[2500]),
    'RATE_OP5': np.array(9*[2500]),
    'RATE_A5': np.array(9*[6000]),
    'RATE_A6': np.array(9*[6000]),
}
np.savez('init_rates.npz', **d)