import sys
from pathlib import Path

# User variables
config_file = "../data/config/closed_loop.yaml"  # Settings of the closed loop (paths in it are relative to the file)
save_folder = "../outputs/closed_loop"           # Output folder; the loop refuses to start if it already exists
run_loop = True  # Set False to skip the loop and only redraw the figures of an existing run in save_folder

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
    return (here / config_file).resolve(), (here / save_folder).resolve()


def run(config, output_dir):
    if run_loop:
        from closed_loop.workflow.run_workflow import main
        main(config, output_dir)  # also draws the figures after every interval
    else:
        from closed_loop.workflow.plot_results import main
        main(output_dir)


if __name__ == '__main__':
    config, output_dir = prepare_context()
    run(config, output_dir)
