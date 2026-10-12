"""Short-term optimization of the daily well controls (Short-term Optimization section, Eq. 6)."""
import numpy as np
from scipy.optimize import minimize


def optimize_controls(predict, u_target, u_max, free, target, tolerance, max_iter, rhobeg):
    """Minimise the relative change in CO2 emission over the daily controls of the `free` wells.

    predict(u) evaluates (n_wells, n_days) controls u with the proxy and returns the interval totals.
    Subject to the constraints, the oil, gas, produced-water and injected-water volumes stay within
    `tolerance` of the target volumes, and the relative NPV change stays smaller than the relative CO2
    change. A volume whose target is zero gets no constraint. COBYLA works on the free controls scaled to
    [0, 1] by their bounds, starting from the target. Returns the optimized controls and the scipy result.
    """
    scale = u_max[free, None]

    def controls(z):
        u = u_target.copy()
        u[free] = np.clip(z, 0, 1).reshape(len(scale), -1) * scale
        return u

    volumes = [key for key in ("oil", "gas", "wp", "wi") if target[key] > 0]
    last = {"z": None}

    def change(z):
        # Relative change from the target. COBYLA evaluates the objective and the constraints separately
        # at each point, so the last proxy evaluation is reused.
        if last["z"] is None or not np.array_equal(z, last["z"]):
            p = predict(controls(z))
            last.update(z=z.copy(), change={k: (p[k] - target[k]) / abs(target[k]) for k in ("co2", "npv", *volumes)})
        return last["change"]

    def constraints(z):
        c = change(z)
        return np.array([tolerance - abs(c[key]) for key in volumes] + [abs(c["co2"]) - abs(c["npv"])])

    z_target = (u_target[free] / scale).ravel()
    result = minimize(lambda z: change(z)["co2"], z_target, method="COBYLA", bounds=[(0, 1)] * z_target.size,
                      constraints={"type": "ineq", "fun": constraints},
                      options={"maxiter": max_iter, "rhobeg": rhobeg})
    return controls(result.x), result
