"""Closed-loop short-term optimization of wind-powered reservoir management (ECMOR 2026).

    python examples/run-closed_loop.py
    python -m closed_loop.workflow.run_workflow data/config/closed_loop.yaml [output_dir]

The loop of Fig. 1 in the paper runs over control intervals of `interval_length` days:

1. Simulate the target controls with the full model. This gives the reference volumes, NPV, power
   demand and emissions of the interval. Optimization is skipped if the target power demand exceeds the
   wind power on every day (no surplus wind to shift demand into) or on no day (no emissions to reduce).
2. Sample control strategies around the target, simulate them and train the per-well RBF proxy.
3. Minimise the CO2 emission predicted by the proxy, subject to the constraints of Eq. (6). With
   `demand_error_std` > 0 the emission is averaged over scenarios of the proxy's error in the daily
   power demand. This keeps the plan from resting on the steps of the emission model, where one gas
   turbine starts or a second one is needed.
4. Simulate the optimized controls with the full model, and apply them only if the emissions went down.
5. Continue the next interval from the final state of the applied strategy.

Everything computed for an interval goes to `<output_dir>/interval_XXX.pkl`, the interval totals to
`<output_dir>/summary.csv`, the log to `<output_dir>/workflow.log` and the figures to `<output_dir>/figures`.
A copy of the configuration is kept as `<output_dir>/config.yaml`.
"""
from pathlib import Path
import logging
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml

from .facility import evaluate
from .optimization import optimize_controls
from .plot_results import plot_cumulative, plot_interval
from .proxy import Proxy, sample_controls
from .simulator import Simulator

TOTALS = {"co2": "CO2", "npv": "NPV", "oil": "oil", "gas": "gas", "wp": "wp", "wi": "wi"}  # key: label
KERNEL_NAMES = {"linear": "linear", "thin_plate_spline": "TPS", "cubic": "cubic", "gaussian": "Gauss",
                "multiquadric": "MQ", "inverse_multiquadric": "IMQ"}
log = logging.getLogger("workflow")


def main(config_file, output_dir=None):
    """Run the closed loop with the settings in `config_file`. `output_dir`, if given, replaces the config's."""
    config_file = Path(config_file).resolve()
    cfg = yaml.safe_load(config_file.read_text())
    path = lambda name: config_file.parent / name
    output_dir = Path(output_dir).resolve() if output_dir is not None else path(cfg["output_dir"])
    if output_dir.exists():
        sys.exit(f"{output_dir} already exists. Delete it or choose another output folder.")
    (output_dir / "figures").mkdir(parents=True)
    shutil.copy(config_file, output_dir / "config.yaml")
    _start_log(output_dir / "workflow.log")

    producers, injectors = cfg["wells"]["producers"], cfg["wells"]["injectors"]
    wells = [*producers, *injectors]
    u_max = np.array([*producers.values(), *injectors.values()], dtype=float)
    injector = np.isin(wells, list(injectors))

    # Daily target controls from the monthly table, clipped to the bounds, and the daily wind power
    days = pd.date_range(cfg["start_date"], cfg["end_date"], freq="D", inclusive="left")
    table = pd.read_csv(path(cfg["target_controls"]), index_col=0, parse_dates=True).to_period("M")
    target_controls = table.loc[days.to_period("M"), [f"Rate{w}" for w in wells]].to_numpy().T
    target_controls = target_controls.clip(0, u_max[:, None])
    wind = np.load(path(cfg["windpower"]))
    if len(wind) < len(days):
        sys.exit(f"{cfg['windpower']} has {len(wind)} days of wind power, the loop needs {len(days)}.")
    wind = wind[:len(days)]

    s = cfg["simulator"]
    sim = Simulator(path(s["deck"]), path(s["model_dir"]), path(s["history_case"]), s["restart_step"],
                    output_dir, producers, injectors, s["flow"], s["n_parallel"])
    rng = np.random.default_rng(cfg["seed"])

    starts = range(0, len(days), cfg["interval_length"])
    std, n_scenarios = cfg["optimization"]["demand_error_std"], cfg["optimization"]["demand_scenarios"]
    log.info(f"Closed loop over {len(starts)} intervals, {days[0]:%d %b %Y} - {days[-1]:%d %b %Y}. "
             + (f"Robust optimization over {n_scenarios} scenarios of a {std} MW error in the daily "
                "power demand." if std > 0 else "Deterministic optimization (Eq. 6)."))
    summary = []
    for i, start in enumerate(starts, start=1):
        window = slice(start, start + cfg["interval_length"])
        dates = days[window]
        log.info("")
        log.info(f"Interval {i}/{len(starts)}: {dates[0]:%d %b %Y} - {dates[-1]:%d %b %Y}")
        clock = time.time()

        record = run_interval(sim, dates, wind[window], target_controls[:, window], wells, u_max, injector, cfg, rng)
        applied = "optimized" if record["status"] == "accepted" else "target"
        sim.accept(applied)

        with open(output_dir / f"interval_{i:03d}.pkl", "wb") as file:
            pickle.dump(record, file)
        summary.append({"interval": i, "start": dates[0].date(), "end": dates[-1].date(), "status": record["status"],
                        **{f"target_{k}": record["target"][k] for k in TOTALS},
                        **{f"applied_{k}": record[applied][k] for k in TOTALS}})
        frame = pd.DataFrame(summary)
        frame.to_csv(output_dir / "summary.csv", index=False)

        _log_interval_totals(record)
        change = {k: 100 * (frame[f"applied_{k}"].sum() / frame[f"target_{k}"].sum() - 1) for k in ("co2", "npv")}
        log.info(f"Interval {i} {record['status']} in {(time.time() - clock) / 60:.1f} min. Cumulative change "
                 f"from target: CO2 {change['co2']:+.2f} %, NPV {change['npv']:+.3f} %")
        try:
            plot_interval(record, i, output_dir / "figures" / f"interval_{i:03d}.png")
            plot_cumulative(frame, output_dir / "figures" / "cumulative.png")
        except Exception as error:  # a figure that fails must not stop the loop
            log.warning(f"Plotting interval {i} failed: {error!r}")

    totals = pd.DataFrame(summary).sum(numeric_only=True)
    scales = (1e3, 1e6, 1e6, 1e6, 1e6, 1e6)
    target, applied = ([totals[f"{s}_{k}"] / c for k, c in zip(TOTALS, scales)] for s in ("target", "applied"))
    log.info("")
    log.info("Totals over all intervals (" + ", ".join(f"{n} {status}" for status, n in
                                                       pd.DataFrame(summary)["status"].value_counts().items()) + ")")
    _table(["", "CO2 [kt]", "NPV [MUSD]", "oil [MSm3]", "gas [MSm3]", "wp [MSm3]", "wi [MSm3]"],
           ["target", *(f"{v:.3f}" for v in target)], ["applied", *(f"{v:.3f}" for v in applied)],
           ["change [%]", *(f"{100 * (a / t - 1):+.2f}" if t else "" for a, t in zip(applied, target))])


def run_interval(sim, dates, wind, u_target, wells, u_max, injector, cfg, rng):
    """Run the closed-loop steps for one interval and return a record of every strategy evaluated.

    `u_target` holds the (n_wells, n_days) target rates. The record's status says what happened: skipped
    (optimization not relevant), optimizer failed, simulation failed, rejected (the full model showed no
    emission reduction) or accepted. The target is applied in every case except accepted.
    """
    economics = cfg["economics"]

    # 1. Target strategy with the full model
    y_target = sim.run([u_target], dates, keep="target")[0]
    if y_target is None:
        raise RuntimeError(f"The target simulation failed, see {sim.work_dir / 'target'}")
    target = evaluate(y_target, wind, economics)
    record = {"dates": dates, "wind": wind, "status": "skipped",
              "target": {"controls": u_target, "response": y_target, **target}}
    deficit = target["power"] > wind
    log.info(f"Target simulated: the wind covers the power demand on {np.sum(~deficit)} of {len(dates)} days")
    if deficit.all() or not deficit.any():
        return record

    # 2. Proxy trained on full-model runs of control strategies sampled around the target
    free = ~injector | (u_target > 0).any(axis=1)  # an injector whose target is zero stays shut
    samples = sample_controls(u_target, u_max, free, cfg["proxy"]["n_samples"], cfg["proxy"]["sample_std"], rng)
    runs = sim.run(samples, dates)
    ok = [n for n, y in enumerate(runs) if y is not None]
    if len(ok) < 2:
        raise RuntimeError(f"Only {len(ok)} of {len(runs)} training simulations succeeded")
    controls = np.concatenate([samples[ok], u_target[None]])  # the target anchors the proxy
    responses = [runs[n] for n in ok] + [y_target]
    n_validation = max(1, round(cfg["proxy"]["validation_fraction"] * len(ok)))
    proxy = Proxy(controls[:, free], responses, [w for w, f in zip(wells, free) if f], u_max[free], n_validation)
    record["training"] = {"controls": controls, "responses": {k: np.array([r[k] for r in responses]) for k in y_target}}
    record["kernels"], record["validation_error"] = proxy.kernels, proxy.validation_error
    log.info(f"Proxy trained on {len(ok)} of {len(runs)} runs")
    _table(["well", *proxy.kernels],
           ["kernel", *(KERNEL_NAMES[k] + ("" if e is None else f"({e:g})") for k, e in proxy.kernels.values())],
           ["val. error [%]", *(f"{100 * e:.2f}" for e in proxy.validation_error.values())])

    # 3. Optimization with the proxy; the wells without a proxy keep their target response. In the robust
    #    case, the target and the candidates are evaluated over the same demand-error scenarios. A scenario
    #    lowers the wind by demand_error_std times one of demand_scenarios equally likely standard-normal
    #    quantiles, which is the same as raising the demand.
    opt = cfg["optimization"]
    quantiles = norm.ppf((np.arange(opt["demand_scenarios"]) + 0.5) / opt["demand_scenarios"])
    scenarios = wind - opt["demand_error_std"] * quantiles[:, None] if opt["demand_error_std"] > 0 else wind
    predict = lambda u: evaluate({**y_target, **proxy(u[free])}, scenarios, economics)
    u_opt, result = optimize_controls(predict, u_target, u_max, free, evaluate(y_target, scenarios, economics),
                                      opt["tolerance"], opt["max_iter"], opt["rhobeg"])
    record["optimizer"] = {"success": result.success, "message": result.message, "nfev": result.nfev}
    log.info(f"COBYLA: {result.message} ({result.nfev} proxy evaluations)")
    if not result.success:
        record["status"] = "optimizer failed"
        return record
    y_proxy = {**y_target, **proxy(u_opt[free])}
    record["proxy"] = {"controls": u_opt, "response": y_proxy, **evaluate(y_proxy, wind, economics)}

    # 4. Optimized strategy with the full model
    y_opt = sim.run([u_opt], dates, keep="optimized")[0]
    if y_opt is None:
        record["status"] = "simulation failed"
        return record
    optimized = evaluate(y_opt, wind, economics)
    record["optimized"] = {"controls": u_opt, "response": y_opt, **optimized}
    record["status"] = "accepted" if optimized["co2"] < target["co2"] else "rejected"
    return record


def _log_interval_totals(record):
    """Table of the interval totals of the target and of the optimum, as predicted and as simulated."""
    target, rows = record["target"], []
    for name, label in (("target", "target"), ("proxy", "predicted"), ("optimized", "simulated")):
        if name in record:
            s = record[name]
            changes = ("" if name == "target" or not target[k] else f"{100 * (s[k] / target[k] - 1):+.2f}" for k in TOTALS)
            rows.append([label, f"{s['co2']:.1f}", f"{s['npv'] / 1e6:.3f}", *changes])
    _table(["", "CO2 [t]", "NPV [MUSD]", *(f"Δ{label} [%]" for label in TOTALS.values())], *rows)


def _table(header, *rows):
    """Log a table framed with box-drawing lines, like the PET logger."""
    widths = [max(10, *(len(row[j]) + 2 for row in (header, *rows))) for j in range(len(header))]
    border = lambda left, middle, right: left + middle.join("─" * w for w in widths) + right
    line = lambda row: "│" + "│".join(f"{cell:^{w}}" for cell, w in zip(row, widths)) + "│"
    for text in (border("┌", "┬", "┐"), line(header), border("├", "┼", "┤"), *map(line, rows), border("└", "┴", "┘")):
        log.info(text)


def _start_log(filename):
    """Log to the console and to `filename`, with time stamps like the PET logger."""
    log.handlers.clear()
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s :  %(message)s", datefmt="%Y-%m-%d│%H:%M:%S")
    for handler in (logging.FileHandler(filename, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        log.addHandler(handler)


if __name__ == "__main__":
    main(*sys.argv[1:])
