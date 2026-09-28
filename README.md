# MultiObjectiveWorkflow

Work-in-progress tools for multi-objective optimization of Drogon-style energy system cases (wind + gas and wind + hydrogen scenarios).

## Install

From the project root:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e .[analysis]
python -m pip install -e .[dev]
```

Note: runtime dependencies used by optimization backends (for example PET-related packages such as `popt`/`subsurface`) are expected to be available in your environment.

## Examples

Both examples run a Pareto sweep over the weights in `objective_weights`: for each weight `w` the optimizer minimizes `w * f1 + (1 - w) * (-NPV)`, where `f1` is total CO2 emissions or CAPEX. Results for each weight go to `<save_folder>/weight_<w>/`, and plots are written after the sweep.

Run the examples from the `examples` directory (the scripts use relative paths):

```bash
cd examples
python run-wind_gas.py
python run-wind_hydrogen.py
```

User-editable variables at the top of each script:
- `energy_sources`: energy sources in the case (checked against the script's case)
- `random_seed`: random seed for reproducibility
- `objective_weights`: weights to run for the Pareto sweep
- `save_folder`: where results are written
- `parallel`: number of simulations to run in parallel
- `run_optimization`: set to `False` to skip optimization and only plot existing results
- `show_plots`: show figures on screen after plotting (figures are saved either way)

### Wind + gas (`run-wind_gas.py`)

Offshore power comes from wind turbines, with gas turbines covering any shortfall. The controls are the time-varying rates of the seven Drogon wells (`RATE_A1`–`RATE_A6`, `RATE_OP5`). The two objectives are NPV and total CO2 emissions from the gas turbines. Emissions are costed in the NPV, and fuel gas is subtracted from exported gas. Config: `data/config/wind_gas.py`. Default output: `outputs/wind_gas/`.

### Wind + hydrogen (`run-wind_hydrogen.py`)

Wind power, with hydrogen storage (electrolysis and fuel cell) covering shortfalls instead of gas turbines. On top of the well rates, it optimizes the number of wind turbines (`N_WT`) and the hydrogen storage capacity (`N_H2`). The two objectives are NPV and CAPEX, and a penalty (EPF) handles hydrogen shortfalls. Config: `data/config/wind_hydrogen.py`. Default output: `outputs/wind_hydrogen/`.

### Plots

Plotting code lives in `examples/plot_results.py`. For each weight, figures are written to `weight_<w>/figures/`:
- `obj_func.png` (or `obj_func_epf.png` with penalty loops): objective history
- `variable_<i>.png`: initial vs. final well-rate controls
- `free_variable_<i>.png` (hydrogen only): initial vs. final `N_WT` / `N_H2`
- `pareto_point.png`:
  - Gas: histograms of NPV and total CO2 across the ensemble
  - Hydrogen: NPV histogram and CAPEX breakdown

In `save_folder` itself:
- `pareto_curve.png`: mean NPV against CO2 (gas) or CAPEX (hydrogen), one point per weight
- `wind_power_profiles.png`: sample wind power profiles

## Tests

```bash
python -m pip install -e .[dev,analysis]
python -m pytest
```

The tests cover `src/facility`, `src/multi_objective_optimization` and `examples/`. They use small synthetic data and stand-ins for PET and OPM Flow, so no reservoir simulations are run.
