import os
import subprocess
from p_tqdm import p_map


def run_command_in_folder(folder):
    # Save the current working directory
    cwd = os.getcwd()
    try:
        os.chdir(folder)
        subprocess.run(["flow", "DROGON.DATA"])
    finally:
        os.chdir(cwd)

<<<<<<< Updated upstream:simulate.py
folders = [f"drogon_ensemble/Drogon_{i}" for i in range(50)]
=======
folders = [f"../../drogon_ensemble/Drogon_{i}" for i in range(50)]
>>>>>>> Stashed changes:multi_objective_optimization/tools/simulate.py

p_map(run_command_in_folder, folders, num_cpus=5)