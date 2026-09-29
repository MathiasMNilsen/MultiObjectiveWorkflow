import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == '__main__':
    
    # StateX to target_controls.csv
    stateX = pd.read_csv('results_NewtonCG/stateX.csv', index_col=None)
    stateX = stateX.to_dict(orient='list')
    report = pd.date_range(start='2020-08-01', end='2025-01-01', freq='MS').to_pydatetime().tolist()

    # Simulate stateX
    from initial_tests.sim_tools import simulate_mako, get_sim_results

    simulate = False
    if simulate:
        simulate_mako(
            filename='../initial_tests/DROGON',
            simfolder='targetSIM',
            **stateX
        )

    # Get sim results
    simres = get_sim_results(
        casename='targetSIM/DROGON',
        datatypes=['WOPR:A1', 'WOPR:A2', 'WOPR:A3', 'WOPR:A4', 'WOPR:OP5', 'WWIR:A5', 'WWIR:A6'],
        dates=report
    )
    # Shift month by one for reporting
    simres.index = simres.index + pd.DateOffset(months=-1)

    # Change name of columns (WOPR:A1 to RateA1, and so on)
    for col in simres.columns:
        new_colname = 'Rate' + col.split(':')[-1]
        simres = simres.rename(columns={col: new_colname})
    print(simres)

    # Save to target_controls.csv
    simres.to_csv('results_NewtonCG/target_controls.csv', index_label='')


