import copy
import importlib.util

import pytest

import plot_results as pr
from conftest import (EXAMPLES_ROOT, GAS_STATES, HYDROGEN_STATES, make_kwens, make_kwsim,
                      write_optimize_results, write_pareto_point)
from multi_objective_optimization import pareto

CASES = {
    'gas': {
        'script': 'run-wind_gas.py',
        'states': GAS_STATES,
        'num_free': 0,
        'hydrogen': False,
        'epf': False,
    },
    'hydrogen': {
        'script': 'run-wind_hydrogen.py',
        'states': HYDROGEN_STATES,
        'num_free': 2,
        'hydrogen': True,
        'epf': True,
    },
}


def load_script(name):
    path = EXAMPLES_ROOT / CASES[name]['script']
    spec = importlib.util.spec_from_file_location(f'run_{name}_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=list(CASES))
def case(request):
    return request.param, CASES[request.param], load_script(request.param)


def test_prepare_context(case, tmp_path, monkeypatch):
    name, _, module = case
    monkeypatch.setattr(module, 'save_folder', str(tmp_path / 'out'))
    kwopt, kwens, kwsim, save_root, project_root = module.prepare_context()
    assert save_root == tmp_path / 'out' and save_root.is_dir()
    assert (project_root / 'examples').is_dir()
    assert kwsim['parallel'] == module.parallel
    assert ('N_WT' in kwens['state']) == (name == 'hydrogen')


def test_prepare_context_rejects_wrong_energy_sources(case, tmp_path, monkeypatch):
    _, _, module = case
    monkeypatch.setattr(module, 'save_folder', str(tmp_path / 'out'))
    monkeypatch.setattr(module, 'energy_sources', ['wind'])
    with pytest.raises(ValueError):
        module.prepare_context()


def test_run_cases_optimizes_each_weight(case, tmp_path, monkeypatch):
    _, _, module = case
    copies, calls = [], []
    monkeypatch.setattr(module, 'copyfile', lambda src, dst: copies.append((src, dst)))
    monkeypatch.setattr(module, 'objective_weights', [0.0, 1.0])
    monkeypatch.setattr(pareto, 'optimize_pareto_point',
                        lambda w, kwopt, kwens, kwsim: calls.append((w, kwopt['save_folder'])))

    module.run_cases({}, {}, {}, tmp_path, EXAMPLES_ROOT.parent)

    assert calls == [(0.0, str(tmp_path / 'weight_0.0')), (1.0, str(tmp_path / 'weight_1.0'))]
    assert all((tmp_path / f'weight_{w}').is_dir() for w in (0.0, 1.0))
    assert copies[0][0].name == 'DROGON.mako'


def make_results(tmp_path, spec, weights):
    for i, weight in enumerate(weights):
        folder = tmp_path / f'weight_{weight}'
        write_optimize_results(folder, len(spec['states']), num_free=spec['num_free'],
                               epf=spec['epf'], seed=i)
        write_pareto_point(folder, co2_level=0.0 if spec['hydrogen'] else 150.0 - 10 * i, seed=i)


def test_plot_all_writes_figures(case, tmp_path, windpower_file, monkeypatch):
    _, spec, module = case
    weights = [0.25, 0.75]
    monkeypatch.setattr(module, 'objective_weights', weights + [0.5])  # 0.5 has no results
    make_results(tmp_path, spec, weights)
    kwens = make_kwens(spec['states'])
    kwsim = make_kwsim(windpower_file, spec['hydrogen'])

    module.plot_all({}, copy.deepcopy(kwens), kwsim, tmp_path, EXAMPLES_ROOT.parent, show=False)

    obj_name = 'obj_func_epf.png' if spec['epf'] else 'obj_func.png'
    for weight in weights:
        figures = tmp_path / f'weight_{weight}' / 'figures'
        assert (figures / obj_name).exists()
        assert (figures / 'pareto_point.png').exists()
        assert all((figures / f'variable_{i}.png').exists() for i in range(len(GAS_STATES)))
    assert not (tmp_path / 'weight_0.5').exists()
    assert (tmp_path / 'pareto_curve.png').exists()
    assert (tmp_path / 'wind_power_profiles.png').exists()
    assert pr.plt.get_fignums() == []


def test_plot_all_show_keeps_figures_open(case, tmp_path, windpower_file, monkeypatch):
    _, spec, module = case
    monkeypatch.setattr(module, 'objective_weights', [0.25])
    make_results(tmp_path, spec, [0.25])
    shown = []
    monkeypatch.setattr(pr.plt, 'show', lambda: shown.append(len(pr.plt.get_fignums())))

    module.plot_all({}, make_kwens(spec['states']), make_kwsim(windpower_file, spec['hydrogen']),
                    tmp_path, EXAMPLES_ROOT.parent, show=True)

    assert len(shown) == 1 and shown[0] > 0
