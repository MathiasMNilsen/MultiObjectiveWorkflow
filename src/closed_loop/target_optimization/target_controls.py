"""Simulates the optimal strategy of the life-cycle optimization and writes its monthly rates, the target of the
closed loop (data/closed_loop/target_controls.csv).

    python examples/run-target_optimization.py   # runs run.py first, then this

Renders the deck with the optimum (stateX.csv) into <run_dir>/targetSIM/, runs OPM Flow there, and writes the
simulated monthly rate of every well to a csv file. The simulated rates, not the controls, are the target, because
a producer can produce less than its control rate when its BHP or water limit binds. The simulation is skipped if
targetSIM/ already holds its results; delete the folder to rerun it.
"""
from pathlib import Path
import os
import subprocess

import pandas as pd
from mako.template import Template
from resdata.summary import Summary

PRODUCERS = ['A1', 'A2', 'A3', 'A4', 'OP5']
INJECTORS = ['A5', 'A6']


def write_target_controls(state_file, deck, model_dir, run_dir, output_file, flow='flow'):
    """Simulate the controls in `state_file` (stateX.csv) and write the monthly well rates to `output_file`."""
    sim_dir = Path(run_dir) / 'targetSIM'

    # Simulate the optimum
    if not (sim_dir / 'DROGON.SMSPEC').exists():
        stateX = pd.read_csv(state_file).to_dict(orient='list')
        sim_dir.mkdir(parents=True, exist_ok=True)
        model_dir = os.path.relpath(Path(model_dir).resolve(), sim_dir.resolve())
        (sim_dir / 'DROGON.DATA').write_text(Template(filename=str(deck)).render(model_dir=model_dir, **stateX))
        subprocess.run([*flow.split(), 'DROGON.DATA'], cwd=sim_dir, check=True)

    # Monthly rates. The rate reported on the first day of a month is the rate of the month before.
    keys = [f'WOPR:{w}' for w in PRODUCERS] + [f'WWIR:{w}' for w in INJECTORS]
    report = pd.date_range(start='2020-08-01', end='2025-01-01', freq='MS').to_pydatetime().tolist()
    rates = Summary(str(sim_dir / 'DROGON')).pandas_frame(time_index=report, column_keys=keys)[keys]
    rates.index = rates.index + pd.DateOffset(months=-1)
    rates.columns = ['Rate' + key.split(':')[-1] for key in keys]
    print(rates)

    # Save to target_controls.csv
    rates.to_csv(output_file, index_label='')
    return rates
