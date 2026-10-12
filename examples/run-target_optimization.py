import sys
from pathlib import Path

# User variables
config_file = "../data/config/target_optimization.yaml"  # PET settings of the life-cycle optimization
save_folder = "../outputs/target_optimization"           # Run folder for the optimization and the target simulation
run_optimization = True  # Set False to skip the optimization and only simulate an existing optimum (stateX.csv)
write_target = True      # Simulate the optimum and write its monthly rates (the target of the closed loop)
flow = "flow"            # OPM Flow command

# Edits below this line is not necessary for typical use cases
# ------------------------------------------------------------


def prepare_context():
    # Make project imports independent of current working directory.
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / 'src'
    for _path in (project_root, src_root):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    here = Path(__file__).resolve().parent
    run_dir = (here / save_folder).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return (here / config_file).resolve(), run_dir, project_root


def run(config, run_dir, project_root):
    import yaml
    deck = project_root / 'data/closed_loop/DROGON_target.mako'
    model_dir = project_root / 'data/drogon_ensemble'

    if run_optimization:
        from closed_loop.target_optimization.run import optimize
        optimize(config, deck, model_dir, run_dir)

    if write_target:
        from closed_loop.target_optimization.target_controls import write_target_controls
        savefolder = yaml.safe_load(config.read_text())['optim']['savefolder']
        output_file = run_dir / 'target_controls.csv'
        write_target_controls(run_dir / savefolder / 'stateX.csv', deck, model_dir, run_dir, output_file, flow)
        print(f"Target written to {output_file}. Copy it to data/closed_loop/target_controls.csv to use it in "
              "the closed loop.")


if __name__ == '__main__':
    config, run_dir, project_root = prepare_context()
    run(config, run_dir, project_root)
