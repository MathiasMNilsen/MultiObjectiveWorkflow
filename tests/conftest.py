import datetime as dt
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless backend, must be set before pyplot is imported

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
EXAMPLES_ROOT = PROJECT_ROOT / 'examples'
for _path in (PROJECT_ROOT, SRC_ROOT, EXAMPLES_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import plot_results  # noqa: E402

GAS_STATES = ['RATE_A1', 'RATE_A2', 'RATE_A3', 'RATE_A4', 'RATE_OP5', 'RATE_A5', 'RATE_A6']
HYDROGEN_STATES = GAS_STATES + ['N_WT', 'N_H2']
NUM_STEPS = 3
REPORT_DATES = [dt.datetime(2020, 7, 1), dt.datetime(2020, 8, 1),
                dt.datetime(2020, 9, 1), dt.datetime(2020, 10, 1)]


@pytest.fixture(autouse=True)
def reset_plot_results():
    """Restore plot_results module globals and close figures after each test."""
    names = ['path_to_files', 'path_to_config', 'path_to_figures', 'results_root',
             '_kwopt', '_kwens', '_kwsim', '_project_root', '_show_figures']
    saved = {name: getattr(plot_results, name) for name in names}
    yield
    for name, value in saved.items():
        setattr(plot_results, name, value)
    plot_results.plt.close('all')


def make_kwens(states):
    kwens = {'ne': 2, 'transform': True, 'state': list(states)}
    for name in states:
        if name == 'N_WT':
            limits = [0, 50]
        elif name == 'N_H2':
            limits = [0, 1000]
        else:
            limits = [0, 4000]
        kwens[f'prior_{name}'] = {'limits': limits}
    return kwens


def make_kwsim(windpower_path, hydrogen=False):
    npv_const = {'wop': 471.7, 'wgp': 0.4, 'wwp': 37, 'wwi': 25, 'wem': 150, 'disc': 0.08}
    if hydrogen:
        npv_const.update({'wt': 1.0e6, 'h2': 2.0e3})
    return {
        'windpower': str(windpower_path),
        'reportpoint': list(REPORT_DATES),
        'npv_const': npv_const,
    }


def write_optimize_results(folder, num_states, num_free=0, num_iter=3, ne=2, epf=False, seed=0):
    """Write optimize_result_*.npz files with normalized state vectors."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    size = (num_states - num_free) * NUM_STEPS + num_free
    for it in range(num_iter):
        x = rng.uniform(0.1, 0.9, size=size)
        fun = -1.0 - it - rng.uniform(0, 0.1, size=ne)
        if epf:
            for inner in range(2):
                np.savez(folder / f'optimize_result_{it}_{inner}.npz', x=x, fun=fun,
                         epf={'penalty': np.full(ne, 1.0 / (it + 1))})
        else:
            np.savez(folder / f'optimize_result_{it}.npz', x=x, fun=fun)


def write_pareto_point(folder, ne=2, ndays=30, npv_level=2.8e9, co2_level=100.0, capex=0.0, seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    npv = npv_level + rng.normal(0, 1e6, size=ne)
    co2 = co2_level + rng.uniform(0, 10, size=(ne, ndays))
    np.savez(folder / 'pareto_point.npz', pred_data=np.array([None]), co2=co2, npv=npv, capex=capex)
    return npv, co2


@pytest.fixture
def windpower_file(tmp_path):
    path = tmp_path / 'wp_ens.npy'
    rng = np.random.default_rng(1)
    np.save(path, rng.uniform(0, 8, size=(4, 120)))
    return path
