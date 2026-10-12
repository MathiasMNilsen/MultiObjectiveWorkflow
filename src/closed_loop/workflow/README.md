# Closed-loop short-term optimization with an RBF proxy

This folder implements the closed-loop workflow of the ECMOR 2026 paper on the Drogon model. The paper is a
follow-up to Nilsen et al. (2025), with the coarse-model proxy replaced by an RBF proxy. Each 15-day control
interval goes through five steps:

1. The target controls from the life-cycle optimization are simulated with the full model (OPM Flow).
   This gives the interval's reference volumes, NPV, power demand and emissions.
2. If there is surplus wind on some days and emissions on others, control strategies are sampled around
   the target. They are simulated with the full model and used to train one RBF interpolant per well.
3. The daily controls are optimized with COBYLA to minimise the CO2 emission predicted by the proxy.
   The constraints keep the interval volumes of oil, gas, produced water and injected water within ε of
   the target, and the relative NPV loss below the relative CO2 reduction (Eq. 6). By default the
   emission is averaged over scenarios of the proxy's power-demand error, which makes the plan robust
   (see Modelling choices).
4. The optimized controls are simulated with the full model. They are applied only if the emissions went
   down.
5. The next interval starts from the final state of the applied strategy.

## Files

| File | Content | Paper |
|---|---|---|
| `run_workflow.py` | The closed loop: reads the configuration and runs the steps above for every interval | Fig. 1 |
| `simulator.py` | Renders the deck, runs OPM Flow in parallel, reads the summary and handles restart files | |
| `facility.py` | Power demand, gas-turbine emissions and interval NPV | Eqs. 2-3 |
| `proxy.py` | Sampling of training strategies and the per-well RBF proxy | Fig. 5 |
| `optimization.py` | The constrained short-term optimization | Eq. 6 |
| `plot_results.py` | Figures of single intervals and of the cumulative change | Figs. 6-9 |

The inputs are outside the package, paths relative to the project root:

| File | Content | Paper |
|---|---|---|
| `data/config/closed_loop.yaml` | All settings; the defaults are those of the paper except `demand_error_std` | Table 1 |
| `data/closed_loop/DROGON_workflow.mako` | OPM Flow deck of Drogon, restarting from a given report step with daily well controls | |
| `data/closed_loop/target_controls.csv` | Monthly target rates from the life-cycle optimization (`../target_optimization`) | Fig. 3 |
| `data/closed_loop/windpower.npy` | Available wind power per day (two 8 MW turbines), starting on 3 Jul 2020 | |
| `data/drogon_ensemble/` | `include/` and `Drogon_Mean/` of the Drogon model, with the restart files of the history run | |

## Running

You need OPM Flow (`flow`) and the `closed-loop` extra (`python -m pip install -e .[closed-loop]`). From the
`examples` folder:

```bash
python run-closed_loop.py   # the closed loop; writes to save_folder (outputs/closed_loop)
```

Set `run_loop = False` in the script to only redraw the figures of an existing run. Single intervals can be
redrawn with `python -m closed_loop.workflow.plot_results outputs/closed_loop 54 78`.

The deck reads the grid and property files from `model_dir`, which needs `include/` and `Drogon_Mean/` from
the Drogon model. The loop starts from report step 82 (1 Jul 2020) of the history run `history_case`
(`data/drogon_ensemble/Drogon_Mean/DROGON.DATA`, which writes its restart files through
`include/schedule/drogon_hist_restart.sch`). Its restart files `DROGON.X0000` and `DROGON.X0082` are
copied to the output folder. With 250 samples per
interval, one interval takes about four minutes at 25 parallel simulations (`simulator.n_parallel`) and
under three minutes at 50.

A run refuses to start if the output folder already exists, so earlier results are never overwritten.

## Output

`output_dir` contains:

- `config.yaml`: a copy of the configuration of the run.
- `workflow.log`: the progress log also shown on the console. Each interval gets a table of the proxy's
  kernel and validation error per well, and a table of the target and optimized totals.
- `summary.csv`: one row per interval, with its status and the totals of the target and the applied
  strategy (`co2` [ton], `npv` [USD], `oil`, `gas`, `wp`, `wi` [Sm3]).
- `interval_XXX.pkl`: a dict with everything computed for the interval:
  - `dates`, `wind`: report dates and available wind power [MW]. The controls for `dates[d]` apply over
    the day that ends on that date.
  - `status`: `skipped`, `optimizer failed`, `simulation failed`, `rejected` or `accepted`. The target is
    applied in every case except `accepted`.
  - `target`, `proxy`, `optimized`: the target, the optimum as predicted by the proxy, and the optimum
    simulated with the full model. Each holds `controls` (n_wells × n_days, producers first, in the order of
    `config.yaml`), `response` (summary key such as `WOPR:A1` → daily values), the daily `power` [MW] and
    `co2_rate` [ton/day], and the interval totals.
  - `training`: the simulated training strategies (`controls`, n_samples × n_wells × n_days) and their
    `responses`. The last sample is the target.
  - `kernels`: the (kernel, epsilon) chosen for each well. `validation_error` holds its RMS error on the
    validation samples as a fraction of the output range. `optimizer` holds the COBYLA status.
- `restarts/`: the final state of every interval, so any interval can be re-simulated.
- `figures/`: `interval_XXX.png` for every interval, and `cumulative.png` with a PDF copy for the paper,
  updated after each interval.

## Modelling choices

- **Proxy.** Each well gets one vector-valued `scipy.interpolate.RBFInterpolator`. It maps the well's
  daily controls, scaled to [0, 1] by the control bounds, to its daily rates, scaled to [0, 1] by their
  training range. A producer predicts its oil, gas and water rates. An injector predicts its injection rate
  and tubing-head pressure, which sets the head of the injection pumps. 10% of the samples choose each
  well's kernel. The choice is linear, thin-plate spline or cubic, or Gaussian, multiquadric or inverse
  multiquadric with shape parameter 0.1-5. The target is always part of the training set, so the proxy is
  exact at the target.
- **Robust optimization.** The emission model has steep steps. A gas turbine that starts burns about
  50 t CO2/day within its first 0.05 MW of load, and a second turbine above 15 MW adds 45 t/day. A
  deterministic optimum puts the demand exactly at these steps on windy days, so a proxy error of a few
  hundredths of a MW can turn a planned emission-free day into a 25-45 t day. With `demand_error_std` σ > 0,
  the optimization minimises the emission averaged over `demand_scenarios` equally likely scenarios
  instead. In each scenario the daily power demand is shifted by σ times a standard-normal quantile, at the
  midpoints of equal-probability slices of the distribution. The shift is implemented as a shift of
  the wind power. The NPV constraint uses the same average, and the target reference is evaluated over the
  same scenarios. The acceptance test and all reported results use the actual wind. σ = 0.05 MW is about the
  proxy's power error at the optimum; σ = 0 gives the deterministic Eq. (6) of the paper.
- **Sampling.** The samples are drawn from a Gaussian around the target with independent components. Each
  component has a standard deviation of 10% of the control range, and the Gaussian is truncated to the
  bounds.
- **Shut injectors.** An injector whose target rate is zero throughout the interval stays shut, in the
  samples and in the optimization. The water-injection constraint is dropped when the target injection is
  zero.
- **Skipped intervals.** The optimization is skipped when the target power demand exceeds the wind power on
  every day, because there is no surplus wind to shift demand into. It is also skipped when the demand
  exceeds the wind power on no day, because there are no emissions to reduce.
- **NPV** in an interval is undiscounted: oil and exported gas are revenue, and water injection, water
  production and CO2 tax are costs. Exported gas is the produced gas minus the turbine fuel.
