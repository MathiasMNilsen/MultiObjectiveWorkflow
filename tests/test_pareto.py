from types import SimpleNamespace

import numpy as np
import pytest

import popt.update_schemes.linesearch as linesearch_module
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


@pytest.fixture
def patched(monkeypatch):
    returns = {'npv': NPV, 'co2': CO2, 'capex': 0.0}
    monkeypatch.setattr(pareto, 'GaussianEnsemble', FakeEnsemble)
    monkeypatch.setattr(pareto, 'flow', lambda kwsim: 'simulator')
    monkeypatch.setattr(pareto, 'objectives',
                        lambda *a, **k: (returns['npv'], returns['co2'], returns['capex']))

    def fake_linesearch(fun, x, jac, hess, args, bounds, **kwopt):
        return SimpleNamespace(x=np.asarray(x) + 0.5)

    monkeypatch.setattr(linesearch_module, 'LineSearch', fake_linesearch)
    return returns


def run(tmp_path, weight=0.5, **kwopt):
    kwopt = {'save_folder': str(tmp_path), **kwopt}
    pareto.optimize_pareto_point(weight, kwopt=kwopt, kwsim={}, kwens={})
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
