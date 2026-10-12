"""Per-well RBF proxy of the full model over one control interval (Proxy Model section, Fig. 5)."""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import truncnorm

# Candidate (kernel, epsilon) pairs. Linear, thin-plate spline and cubic kernels are scale invariant, so
# only the other kernels need a range of shape parameters.
KERNELS = [("linear", None), ("thin_plate_spline", None), ("cubic", None)] + [
    (kernel, epsilon) for kernel in ("gaussian", "multiquadric", "inverse_multiquadric")
    for epsilon in (0.1, 0.5, 1.0, 2.0, 5.0)]


def sample_controls(u_target, u_max, free, n_samples, std, rng):
    """Draw control schedules from a Gaussian around the target, truncated to the bounds [0, u_max].

    The standard deviation of every daily control is `std` times its range, and the draws are independent.
    Wells that are not `free` keep their target controls.
    """
    z = u_target / u_max[:, None]
    z = truncnorm.rvs(-z / std, (1 - z) / std, loc=z, scale=std, size=(n_samples, *z.shape), random_state=rng)
    samples = z * u_max[:, None]
    samples[:, ~free] = u_target[~free]
    return samples


class Proxy:
    """One vector-valued RBF interpolant per well, mapping the well's daily controls to its daily rates.

    A producer predicts its oil, gas and water rates. An injector predicts its injection rate and
    tubing-head pressure. The inputs are scaled to the unit cube by the control bounds, and the outputs to
    [0, 1] by their range in the training data. The first `n_validation` samples pick each well's kernel,
    and the chosen kernel is then refitted to all samples. `kernels` holds the (kernel, epsilon) of each
    well, and `validation_error` its RMS error on the validation samples as a fraction of the output range.
    """

    def __init__(self, controls, responses, wells, u_max, n_validation):
        """controls: (n_samples, n_wells, n_days) array of rates, responses: the matching simulator outputs."""
        self.u_max, self.models, self.kernels, self.validation_error = u_max, [], {}, {}
        for well, x, x_max in zip(wells, controls.transpose(1, 0, 2), u_max):
            keys = [key for key in responses[0] if key.endswith(f":{well}")]
            x = x / x_max
            y = np.array([np.concatenate([r[key] for key in keys]) for r in responses])
            y_min, y_range = y.min(axis=0), np.ptp(y, axis=0)
            y_range[y_range == 0] = 1.0
            y = (y - y_min) / y_range
            errors = {candidate: _validation_error(x, y, n_validation, *candidate) for candidate in KERNELS}
            kernel, epsilon = min(errors, key=errors.get)
            self.models.append((keys, RBFInterpolator(x, y, kernel=kernel, epsilon=epsilon), y_min, y_range))
            self.kernels[well], self.validation_error[well] = (kernel, epsilon), np.sqrt(errors[kernel, epsilon])

    def __call__(self, u):
        """Predict the summary vectors of the proxy wells for their (n_wells, n_days) controls `u`."""
        response = {}
        for (keys, rbf, y_min, y_range), x, x_max in zip(self.models, u, self.u_max):
            y = np.maximum(rbf(x[None] / x_max)[0] * y_range + y_min, 0.0)  # rates and pressures are >= 0
            response.update(zip(keys, y.reshape(len(keys), -1)))
        return response


def _validation_error(x, y, n_validation, kernel, epsilon):
    """Mean squared error on the first `n_validation` samples of an RBF fitted to the remaining ones."""
    try:
        rbf = RBFInterpolator(x[n_validation:], y[n_validation:], kernel=kernel, epsilon=epsilon)
    except (np.linalg.LinAlgError, ValueError):  # singular system, or too few samples for the kernel
        return np.inf
    return np.mean((rbf(x[:n_validation]) - y[:n_validation]) ** 2)
