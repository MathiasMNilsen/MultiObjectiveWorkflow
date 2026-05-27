from shutil import copyfile
import numpy as np
import datetime as dt

from input_output import read_config

from multi_objective_optimization.tools.pareto import optimize_pareto_point

np.random.seed(26112025)

copyfile('../init/DROGON.mako' ,'./DROGON.mako')

for w in [0.6, 0.7, 0.8]:

    # Define save folder
    s = f"../results/run2w{w}"
    with open('config.yaml', 'r') as f:
        content = f.read()
    new_content = content.replace('save_folder: ""', f'save_folder: "{s}"')
    with open('config_pareto.yaml', 'w') as f:
        f.write(new_content)

    # Run optimization for given weight
    optimize_pareto_point(w)
