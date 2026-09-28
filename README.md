# MultiObjectiveWorkflow

Work-in-progress tools for multi-objective optimization of Drogon-style energy system cases (currently focused on wind + hydrogen scenarios).

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

## Current example

`examples/run-wind_hydrogen.py` runs a Pareto sweep for the weights in `objective_weights` and writes results to `save_folder`.

User-editable variables in that script:
- `energy_sources`
- `random_seed`
- `objective_weights`
- `save_folder`

Run it from the `examples` directory (the script uses relative paths):

```bash
cd examples
python run-wind_hydrogen.py
```

Current debug setting in the example uses a small ensemble (`kwens['ne'] = 2`, `kwens['num_models'] = 2`) for quick checks.

