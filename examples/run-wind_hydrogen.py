from shutil import copyfile
import numpy as np
import datetime as dt
import os
import sys
from pathlib import Path
import plot_results as pr

# User variables
energy_sources = ['wind', 'hydrogen']  # Energy sources to optimize
random_seed = int(dt.datetime.now().strftime("%d%m%Y"))  # Random seed for reproducibility
objective_weights = [0.5, 0.9, 0.99]  # Multi-objective weights
save_folder = "../outputs/wind_hydrogen"  # Save folder for results
parallel = 8  # Number of parallel simulations to run
run_optimization = False  # Set False to skip optimization and only plot existing results
show_plots = True  # Whether to show plots after optimization

# Edits below this line is not necessary for typical use cases
# ------------------------------------------------------------

def prepare_context():
    # Make project imports independent of current working directory.
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / 'src'
    for _path in (project_root, src_root):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    # Set random seed early
    np.random.seed(random_seed)

    save_root = (Path(__file__).resolve().parent / save_folder).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    # Load default options
    if 'wind' in energy_sources and 'hydrogen' in energy_sources:
        print("Running optimization for wind and hydrogen energy sources.")
        from data.config.wind_hydrogen import kwopt, kwens, kwsim
    else:
        raise ValueError("This script requires both wind and hydrogen in energy_sources.")

    # Set user options
    kwsim['parallel'] = parallel

    return kwopt, kwens, kwsim, save_root, project_root

def run_cases(kwopt, kwens, kwsim, save_root, project_root):
    from multi_objective_optimization.pareto import optimize_pareto_point

    copyfile(project_root / 'data/init/DROGON.mako', Path(__file__).resolve().parent / 'DROGON.mako')

    for w in objective_weights:
        output_dir = save_root / f"weight_{w}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set save folder
        kwopt['save_folder'] = str(output_dir)

        # Run optimization for given weight
        optimize_pareto_point(w, kwopt=kwopt, kwens=kwens, kwsim=kwsim)

def plot_all(kwopt, kwens, kwsim, results_root, project_root, show=False):

    for weight in objective_weights:
        path_to_files = results_root / f'weight_{weight}'
        if not path_to_files.exists():
            continue
        path_to_figures = path_to_files / 'figures'
        path_to_figures.mkdir(exist_ok=True)

        pr.set_plot_context(
            path_to_files_=str(path_to_files) + os.sep,
            path_to_figures_=str(path_to_figures) + os.sep,
            results_root_=results_root,
            kwopt=kwopt,
            kwens=kwens,
            kwsim=kwsim,
            project_root=project_root,
            show_figures=show,
        )

        # Load reference values (for scaling) and actual state
        final_data_path = path_to_files / 'final_data.npz'
        if final_data_path.exists():
            final_data = np.load(final_data_path, allow_pickle=True)
            final_data = final_data['final_data'][()]
        else:
            final_data = None

        pr.plot_obj_func(2845473526.0)
        pr.plot_state([1, 1, 1, 1, 1, 1, 1], order='C', state_actual=final_data)
        pr.plot_pareto_point_hydrogen()

    pr.set_plot_context(results_root_=results_root, kwopt=kwopt, kwens=kwens, kwsim=kwsim, project_root=project_root,
                         show_figures=show)
    pr.plot_pareto_curve(selected_weights=objective_weights)
    pr.plot_wind_power_profiles()

    if show:
        pr.plt.show()
        
if __name__ == '__main__':
    kwopt, kwens, kwsim, results_root, project_root = prepare_context()
    if run_optimization:
        run_cases(kwopt, kwens, kwsim, results_root, project_root)
    plot_all(kwopt, kwens, kwsim, results_root, project_root, show_plots)
