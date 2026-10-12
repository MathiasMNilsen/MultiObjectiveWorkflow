import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multi_objective_optimization import pareto

NPV = np.array([2.0e9, 3.0e9])
CO2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class FakeEnsemble:
    """Stands in for GaussianEnsemble: records the objective and forwards function calls to obj_func."""
    last = None

    def __init__(self, kwens, sim, objective):
        self.objective = objective
        self.obj_func = objective
        self.calls = []
        FakeEnsemble.last = self

    def get_state(self):
        return np.zeros(3)

    def get_cov(self):
        return 2.0 * np.eye(3)

    def get_bounds(self):
        return [(0, 1)] * 3

    def function(self, x, *args, **kwargs):
        self.calls.append(np.array(x))
        return self.obj_func([{}], {}, (None, None), **kwargs)


class FakeOptimizer:
    """Stands in for PET's LineSearch/EnOpt: records its options and 'optimizes' by adding 0.5 to x0."""
    last = None

    def __init__(self, x0, fun, jac=None, hess=None, args=(), bounds=None, callback=None, **options):
        self.x0 = np.asarray(x0)
        self.jac = jac
        self.hess = hess
        self.args = args
        self.bounds = bounds
        self.callback = callback
        self.options = options
        self.epf = options.get('epf', {})
        FakeOptimizer.last = self

    def run_optimization(self):
        self.optimize_results = {'x': self.x0 + 0.5}


@pytest.fixture
def patched(monkeypatch):
    returns = {'npv': NPV, 'co2': CO2, 'capex': 0.0}
    monkeypatch.setattr(pareto, 'GaussianEnsemble', FakeEnsemble)
    monkeypatch.setattr(pareto, 'flow', lambda kwsim: 'simulator')
    monkeypatch.setattr(pareto, 'objectives',
                        lambda *a, **k: (returns['npv'], returns['co2'], returns['capex']))
    monkeypatch.setattr(pareto, 'LineSearch', FakeOptimizer)
    monkeypatch.setattr(pareto, 'EnOpt', FakeOptimizer)
    return returns


def run(tmp_path, weight=0.5, kwens=None, **kwopt):
    kwopt = {'save_folder': str(tmp_path), **kwopt}
    pareto.optimize_pareto_point(weight, kwopt=kwopt, kwsim={}, kwens=kwens or {})
    return FakeEnsemble.last


def test_final_evaluation_saves_pareto_point(tmp_path, patched):
    ens = run(tmp_path)
    np.testing.assert_allclose(ens.calls[-1], 0.5)  # evaluated at the optimizer's x
    data = np.load(tmp_path / 'pareto_point.npz', allow_pickle=True)
    np.testing.assert_allclose(data['npv'], NPV)
    np.testing.assert_allclose(data['co2'], CO2)
    assert float(data['capex']) == 0.0


@pytest.mark.parametrize('weight', [0.0, 0.25, 1.0])
def test_weighted_sum_uses_co2_and_npv(tmp_path, patched, weight):
    ens = run(tmp_path, weight=weight, obj_scaling=[10.0, 1.0e9, 1.0])
    value = ens.objective([{}], {}, (None, None))
    expected = weight * CO2.sum(axis=1) / 10.0 - (1 - weight) * NPV / 1.0e9
    np.testing.assert_allclose(value, expected)


def test_weighted_sum_scalar_scaling(tmp_path, patched):
    ens = run(tmp_path, weight=0.5, obj_scaling=2.0)
    value = ens.objective([{}], {}, (None, None))
    np.testing.assert_allclose(value, 0.5 * CO2.sum(axis=1) / 2.0 - 0.5 * NPV / 2.0)


def test_weighted_sum_falls_back_to_capex_without_co2(tmp_path, patched):
    patched['co2'] = np.zeros_like(CO2)
    patched['capex'] = np.array([5.0, 7.0])
    ens = run(tmp_path, weight=1.0)
    np.testing.assert_allclose(ens.objective([{}], {}, (None, None)), [5.0, 7.0])


def test_weighted_sum_adds_scaled_penalty(tmp_path, patched):
    ens = run(tmp_path, weight=0.0, obj_scaling=[1.0, 1.0, 4.0])
    epf = {'penalty': [np.array([4.0, 8.0]), np.array([4.0, 0.0])]}
    value = ens.objective([{}], {}, (None, None), epf=epf)
    np.testing.assert_allclose(value, -NPV + np.array([2.0, 2.0]))


def test_weighted_sum_does_not_save_during_optimization(tmp_path, patched):
    ens = run(tmp_path)
    (tmp_path / 'pareto_point.npz').unlink()
    ens.objective([{}], {}, (None, None))
    assert not (tmp_path / 'pareto_point.npz').exists()


def test_unknown_method_raises(tmp_path, patched):
    with pytest.raises(ValueError, match='Unknown main_method'):
        run(tmp_path, main_method='Nope')


# ---------------------------------------------------------------- optimizer options

def test_linesearch_options(tmp_path, patched):
    run(tmp_path, kwens={'transform': True}, method='Newton-CG', obj_scaling=2.0, maxiter=3)
    opt = FakeOptimizer.last
    # PET's own transform stays off; pareto scales to the unit cube itself
    assert opt.options == {'save_folder': str(tmp_path), 'method': 'Newton-CG', 'maxiter': 3,
                           'transform': False, 'saveit': False}
    np.testing.assert_allclose(opt.args[0], 2.0 * np.eye(3))  # the covariance goes first in args


def test_enopt_options_copy_epf(tmp_path, patched):
    epf = {'r': 0.1, 'penalty': [np.zeros(2)]}
    run(tmp_path, main_method='EnOpt', method='GD', epf=epf)
    opt = FakeOptimizer.last
    assert opt.options['method'] == 'GD' and 'main_method' not in opt.options
    # kwopt is reused for every weight, so the optimizer (and the final evaluation) work on a copy
    assert opt.options['epf'] is not epf and opt.options['epf']['r'] == 0.1
    assert isinstance(epf['penalty'], list)


# ---------------------------------------------------------------- unit-cube scaling

class FakeScaledEnsemble(FakeEnsemble):
    """Controls in [0, 10] with variance 4; gradient and Hessian as the natural ensemble estimates would scale."""

    def get_state(self):
        return np.full(3, 2.0)

    def get_cov(self):
        return 4.0 * np.eye(3)

    def get_bounds(self):
        return [(0, 10)] * 3

    def gradient(self, x, cov, **kwargs):
        self.grad_args = (np.array(x), np.array(cov))
        return np.full(3, 10.0)

    def hessian(self, x, cov, **kwargs):
        return np.full((3, 3), 100.0)


def test_transform_scales_to_unit_cube(tmp_path, patched, monkeypatch):
    monkeypatch.setattr(pareto, 'GaussianEnsemble', FakeScaledEnsemble)
    ens = run(tmp_path, kwens={'transform': True})
    opt = FakeOptimizer.last
    np.testing.assert_allclose(opt.x0, 0.2)
    np.testing.assert_allclose(opt.args[0], 0.04 * np.eye(3))
    np.testing.assert_allclose(ens.calls[-1], 7.0)  # optimum u = 0.7 evaluated at x = 10 * 0.7

    # The optimizer's callables work in u, the ensemble in x
    np.testing.assert_allclose(opt.jac(np.full(3, 0.5), opt.args[0]), 1.0)
    x, cov = ens.grad_args
    np.testing.assert_allclose(x, 5.0)
    np.testing.assert_allclose(cov, 4.0 * np.eye(3))
    np.testing.assert_allclose(opt.hess(np.full(3, 0.5), opt.args[0]), np.eye(3))  # diagonal for Newton-CG


def test_no_transform_keeps_original_units(tmp_path, patched, monkeypatch):
    monkeypatch.setattr(pareto, 'GaussianEnsemble', FakeScaledEnsemble)
    run(tmp_path)
    opt = FakeOptimizer.last
    np.testing.assert_allclose(opt.x0, 2.0)
    np.testing.assert_allclose(opt.args[0], 4.0 * np.eye(3))
    assert opt.bounds == [(0, 10)] * 3


# ---------------------------------------------------------------- PET frame -> pred_data

def test_to_pred_data_converts_frame():
    dates = [dt.datetime(2020, 7, 1), dt.datetime(2020, 8, 1)]
    frame = pd.DataFrame({'fopt': [np.array([0.0, 0.0]), np.array([10.0, 20.0])],
                          'wthp:a5': [np.array([50.0, 51.0]), np.array([52.0, 53.0])]}, index=dates, dtype=object)
    pred_data = pareto.to_pred_data(frame)
    assert len(pred_data) == 2 and set(pred_data[1]) == {'fopt', 'wthp:a5'}
    assert pred_data[1]['fopt'].shape == (1, 2)
    np.testing.assert_allclose(pred_data[1]['fopt'], [[10.0, 20.0]])


def test_to_pred_data_single_member():
    frame = pd.DataFrame({'fopt': [np.float64(3.0)]}, dtype=object)
    assert pareto.to_pred_data(frame)[0]['fopt'].shape == (1, 1)


def test_to_pred_data_passes_lists_through():
    pred_data = [{'fopt': np.zeros((1, 2))}]
    assert pareto.to_pred_data(pred_data) is pred_data


# ---------------------------------------------------------------- saving iterations

def optimizer_state(iteration, epf=None, epf_iteration=1):
    return SimpleNamespace(iteration=iteration, epf=epf or {}, epf_iteration=epf_iteration,
                           xk=np.full(3, 0.2 * iteration), fk=np.full(2, -1.0 - iteration),
                           xk_old=np.full(3, 0.2 * (iteration - 1)), fk_old=np.full(2, -float(iteration)))


def test_save_iteration_plain(tmp_path):
    pareto.save_iteration(optimizer_state(1), str(tmp_path))
    pareto.save_iteration(optimizer_state(2), str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == [f'optimize_result_{i}.npz' for i in range(3)]
    first = np.load(tmp_path / 'optimize_result_0.npz')
    np.testing.assert_allclose(first['x'], 0.0)
    np.testing.assert_allclose(first['fun'], -1.0)


def test_save_iteration_epf_counts_outer_loops_from_zero(tmp_path):
    epf = {'r': 1.0, 'penalty': np.array([0.5, 0.5])}
    pareto.save_iteration(optimizer_state(1, epf, epf_iteration=1), str(tmp_path))
    pareto.save_iteration(optimizer_state(1, epf, epf_iteration=2), str(tmp_path))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ['optimize_result_0_0.npz', 'optimize_result_0_1.npz',
                     'optimize_result_1_0.npz', 'optimize_result_1_1.npz']
    saved = np.load(tmp_path / 'optimize_result_1_1.npz', allow_pickle=True)['epf'][()]
    np.testing.assert_allclose(saved['penalty'], [0.5, 0.5])
