"""Static Python config mirrored from wind-gas/config.yaml."""

import datetime as dt


def _monthly_reportpoints(start: dt.datetime, end: dt.datetime):
    """Build month-start reportpoints including both start and end dates."""
    points = []
    current = start
    while current <= end:
        points.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return points


kwopt = {
    'maxiter': 20,
    'method': 'Newton-CG',
    'step_size_adapt': 2,
    'obj_scaling': [187583, 2845473526],
    'save_folder': '',
}

kwsim = {
    'parallel': 8,
    'sim_limit': 24000,
    'runfile': 'DROGON',
    'windpower': '../data/init/wp_ens.npy',
    'reporttype': 'dates',
    'datatype': ['fopt', 'fgpt', 'fwpt', 'fwit', 'wthp:a5', 'wthp:a6'],
    'simoptions': {
        'mpi': 'mpirun -np 5',
        'sim_flag': '--parsing-strictness=low',
    },
    'reportpoint': _monthly_reportpoints(
        dt.datetime(2020, 7, 1),
        dt.datetime(2025, 1, 1),
    ),
    'npv_const': {
        'wop': 471.7,
        'wgp': 0.4,
        'wwp': 37,
        'wwi': 25,
        'wem': 150,
        'disc': 0.08,
    },
}

kwens = {
    'ne': 50,
    'transform': True,
    'num_models': 50,
    'state': ['RATE_A1', 'RATE_A2', 'RATE_A3', 'RATE_A4', 'RATE_OP5', 'RATE_A5', 'RATE_A6'],
    'prior_RATE_A1': {
        'mean': '../data/init/rateA1.npz',
        'var': 22500,
        'limits': [0, 3000],
    },
    'prior_RATE_A2': {
        'mean': '../data/init/rateA2.npz',
        'var': 40000,
        'limits': [0, 4000],
    },
    'prior_RATE_A3': {
        'mean': '../data/init/rateA3.npz',
        'var': 40000,
        'limits': [0, 4000],
    },
    'prior_RATE_A4': {
        'mean': '../data/init/rateA4.npz',
        'var': 40000,
        'limits': [0, 4000],
    },
    'prior_RATE_OP5': {
        'mean': '../data/init/rateOP5.npz',
        'var': 22500,
        'limits': [0, 3000],
    },
    'prior_RATE_A5': {
        'mean': '../data/init/rateA5.npz',
        'var': 160000,
        'limits': [0, 8000],
    },
    'prior_RATE_A6': {
        'mean': '../data/init/rateA6.npz',
        'var': 160000,
        'limits': [0, 8000],
    },
}

