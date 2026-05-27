from shutil import copyfile
import numpy as np

from multi_objective_optimization.tools.pareto import optimize_pareto_point

np.random.seed(26112025)

copyfile('../init/DROGON.mako' ,'./DROGON.mako')

for w in [0.5]:

    # Define save folder
    s = f"../results/run2eval"
    with open('config.yaml', 'r') as f:
        content = f.read()
    new_content = content.replace('save_folder: ""', f'save_folder: "{s}"')
    new_content = new_content.replace('num_models: 50', 'num_models: 1')
    new_content = new_content.replace('maxiter: 30', 'maxiter: 0')
    new_content = new_content.replace('#mpi: mpirun -np 5', 'mpi: mpirun -np 5')
    new_content += '\n  del_folder: false'

    with open('config_pareto.yaml', 'w') as f:
        f.write(new_content)

    # Run optimization for given weight
    # optimize_pareto_point(w)

