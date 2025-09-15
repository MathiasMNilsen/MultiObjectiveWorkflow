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

folders = [f"Drogon_{i}" for i in range(50)]

p_map(run_command_in_folder, folders, num_cpus=5)