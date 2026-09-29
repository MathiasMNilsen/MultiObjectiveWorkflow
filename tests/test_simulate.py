import importlib
import os
import sys
import types

import pytest

MODULE = 'multi_objective_optimization.simulate'


@pytest.fixture
def simulate(monkeypatch):
    """Import simulate.py with p_map stubbed, since the module launches simulations at import time."""
    calls = []
    fake_p_tqdm = types.ModuleType('p_tqdm')
    fake_p_tqdm.p_map = lambda func, items, **kwargs: calls.append((func, list(items), kwargs))
    monkeypatch.setitem(sys.modules, 'p_tqdm', fake_p_tqdm)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    module = importlib.import_module(MODULE)
    yield module, calls
    sys.modules.pop(MODULE, None)


def test_import_dispatches_all_ensemble_folders(simulate):
    module, calls = simulate
    assert len(calls) == 1
    func, folders, kwargs = calls[0]
    assert func is module.run_command_in_folder
    assert folders == [f'../../drogon_ensemble/Drogon_{i}' for i in range(50)]
    assert kwargs == {'num_cpus': 5}


def test_run_command_in_folder_runs_flow_there(simulate, tmp_path, monkeypatch):
    module, _ = simulate
    seen = []
    monkeypatch.setattr(module.subprocess, 'run', lambda cmd: seen.append((cmd, os.getcwd())))
    cwd = os.getcwd()
    module.run_command_in_folder(str(tmp_path))
    assert seen == [(['flow', 'DROGON.DATA'], str(tmp_path))]
    assert os.getcwd() == cwd


def test_run_command_in_folder_restores_cwd_on_error(simulate, tmp_path, monkeypatch):
    module, _ = simulate

    def boom(cmd):
        raise RuntimeError('flow failed')

    monkeypatch.setattr(module.subprocess, 'run', boom)
    cwd = os.getcwd()
    with pytest.raises(RuntimeError):
        module.run_command_in_folder(str(tmp_path))
    assert os.getcwd() == cwd
