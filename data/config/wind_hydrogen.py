"""Static Python config mirrored from wind-hydrogen_ecmor26/config.yaml."""

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
    'maxiter': 30,
    'main_method': 'EnOpt',
    'method': 'GD',
    'step_size_adapt': 0,
    'lsmethod': 0,
    'lsmaxiter': 5,
    'alpha': 0.15,
    'ftol': 0.01,
    'resample': 5,
    'normalize': True,
    'epf': {
        'r': 0.1,
        'max_epf_iter': 5,
        'r_factor': 10,
        'tol_factor': 0.5,
        'conv_crit': 1.0e-9,
    },
    'savedata': ['epf'],
    'save_folder': '',
    'obj_scaling': 2845473526.0,
}

kwsim = {
    'parallel': 8,
    'sim_limit': 24000,
    'runfile': 'DROGON',
    'windpower': '../data/init/wp_ens.npy',
    'reporttype': 'dates',
    'datatype': ['fopt', 'fgpt', 'fwpt', 'fwit', 'wthp:a5', 'wthp:a6'],
    'simoptions': {
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
        'n_gt': 0,
        'wt': 7200000.0,
        'h2': 180000.0,
    },
}

kwens = {
    'ne': 50,
    'transform': True,
    'num_models': 50,
    'state': ['RATE_A1', 'RATE_A2', 'RATE_A3', 'RATE_A4', 'RATE_OP5', 'RATE_A5', 'RATE_A6', 'N_WT', 'N_H2'],
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
    'prior_N_WT': {
        'mean': 35,
        'var': 10,
        'limits': [0, 40],
    },
    'prior_N_H2': {
        'mean': 350,
        'var': 100,
        'limits': [0, 400],
    },
}

