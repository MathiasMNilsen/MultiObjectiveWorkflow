import numpy as np
from shutil import copyfile

from input_output import read_config
from multi_objective_optimization.tools.pareto import optimize_pareto_point

copyfile('../init/DROGON.mako' ,'./DROGON.mako')

for w in [0.0, 0.25, 0.5, 0.75, 1.0]:

    # Define save folder
    s = f"../results/wind-gas_weight_{w}"
    with open('config.yaml', 'r') as f:
        content = f.read()
    new_content = content.replace('save_folder: ""', f'save_folder: "{s}"')
    with open('config_pareto.yaml', 'w') as f:
        f.write(new_content)

    # Set random seed and run
    np.random.seed(29_11_1997)
    optimize_pareto_point(w)