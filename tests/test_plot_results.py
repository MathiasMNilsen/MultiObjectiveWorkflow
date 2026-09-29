import os

import numpy as np
import pandas as pd
import pytest
import yaml

import plot_results as pr
from conftest import (GAS_STATES, HYDROGEN_STATES, NUM_STEPS, REPORT_DATES, make_kwens,
                      make_kwsim, write_optimize_results, write_pareto_point)


def set_context(folder, kwens, kwsim, kwopt=None, results_root=None, show=False):
    figures = folder / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    pr.set_plot_context(
        path_to_files_=str(folder) + os.sep,
        path_to_figures_=str(figures) + os.sep,
        results_root_=results_root or folder.parent,
        kwopt=kwopt or {},
        kwens=kwens,
        kwsim=kwsim,
        show_figures=show,
    )
    return figures


# ---------------------------------------------------------------- small helpers

@pytest.mark.parametrize('name, expected', [
    ('optimize_result_3.npz', (3, None)),
    ('optimize_result_2_5.npz', (2, 5)),
    ('optimize_result_a.npz', None),
    ('optimize_result_1_2_3.npz', None),
    ('pareto_point.npz', None),
])
def test_parse_result_indices(name, expected):
    assert pr._parse_result_indices(name) == expected


@pytest.mark.parametrize('label, expected', [
    ('Rated wind capacity (MW)', ('Rated wind capacity', 'MW')),
    ('RATE_A1 (Sm3/day)', ('RATE_A1', 'Sm3/day')),
    ('No unit', ('No unit', '')),
    (None, ('', '')),
])
def test_split_label_and_unit(label, expected):
    assert pr._split_label_and_unit(label) == expected


def test_inverse_transform():
    np.testing.assert_allclose(pr._inverse_transform(np.array([0.0, 0.5, 1.0]), 10, 20), [10, 15, 20])
    np.testing.assert_allclose(pr._inverse_transform(np.array([0.3, 0.7]), 5, 5), [5, 5])


def test_collapse_duplicate_x_averages_and_sorts():
    x, y = pr._collapse_duplicate_x([3, 1, 1, 2], [30, 10, 20, 5])
    np.testing.assert_allclose(x, [1, 2, 3])
    np.testing.assert_allclose(y, [15, 5, 30])


def test_least_squares_curve_recovers_quadratic():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2 * x ** 2 - x + 1
    dense_x, dense_y = pr._least_squares_curve_eval(x, y, num=50)
    assert dense_x.size == 50
    np.testing.assert_allclose(dense_y, 2 * dense_x ** 2 - dense_x + 1, atol=1e-8)


def test_least_squares_curve_single_point():
    x, y = pr._least_squares_curve_eval([1.0], [2.0])
    np.testing.assert_allclose(x, [1.0])
    np.testing.assert_allclose(y, [2.0])


def test_set_axis_padding_anchor_zero():
    fig, ax = pr.plt.subplots()
    pr._set_axis_padding(ax, [2.0, 4.0], anchor_zero=True)
    assert ax.get_ylim()[0] == 0.0
    pr._set_axis_padding(ax, [2.0, 4.0])
    assert ax.get_ylim()[0] < 2.0 and ax.get_ylim()[1] > 4.0


# ---------------------------------------------------------------- config loaders

def test_load_state_names_pads_and_truncates():
    pr.set_plot_context(kwens={'state': ['A', 'B']})
    assert pr._load_state_names(None, 3) == ['A', 'B', None]
    assert pr._load_state_names(None, 1) == ['A']


def test_load_state_limits_from_kwens():
    pr.set_plot_context(kwens=make_kwens(['RATE_A1', 'N_WT', 'MISSING']))
    pr._kwens.pop('prior_MISSING')
    limits, transform = pr._load_state_limits(None, ['RATE_A1', 'N_WT', 'MISSING'])
    assert transform is True
    assert limits == [(0.0, 4000.0), (0.0, 50.0), (0.0, 1.0)]


def test_load_state_limits_without_transform():
    pr.set_plot_context(kwens={'state': ['A'], 'transform': False})
    assert pr._load_state_limits(None, ['A']) == ([], False)


def test_load_state_names_and_limits_from_yaml(tmp_path):
    pr._kwens = None
    cfg = {'ensemble': {'state': ['A', 'B'], 'prior_A': {'limits': [1, 3]}}}
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(cfg))
    assert pr._load_state_names(str(tmp_path), 2) == ['A', 'B']
    limits, transform = pr._load_state_limits(str(tmp_path), ['A', 'B'])
    assert transform is True
    assert limits == [(1.0, 3.0), (0.0, 1.0)]


def test_load_time_index_from_reportpoint_list():
    pr.set_plot_context(kwsim={'reportpoint': REPORT_DATES})
    index = pr._load_time_index(None, NUM_STEPS)
    assert list(index) == [pd.Timestamp(d) for d in REPORT_DATES]
    assert pr._load_time_index(None, NUM_STEPS + 1) is None


def test_load_time_index_from_start_end_dict():
    pr.set_plot_context(kwsim={'reportpoint': {'start': '2020-01-01', 'end': '2020-01-05'}})
    index = pr._load_time_index(None, 4)
    assert len(index) == 5
    assert index[-1] == pd.Timestamp('2020-01-05')


# ---------------------------------------------------------------- figure handling

@pytest.mark.parametrize('show, expected_open', [(False, 0), (True, 1)])
def test_close_figure_respects_show_flag(show, expected_open):
    pr.set_plot_context(show_figures=show)
    fig, _ = pr.plt.subplots()
    pr._close_figure(fig)
    assert len(pr.plt.get_fignums()) == expected_open


def test_plot_distribution_marks_mean_and_percentiles():
    fig, ax = pr.plt.subplots()
    values = np.arange(1.0, 101.0)
    pr._plot_distribution(ax, values, 'Title', 'Unit')
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ['Title', 'Mean: 50.5', 'P10: 10.9', 'P90: 90.1']
    assert ax.get_title() == 'Title' and ax.get_xlabel() == 'Unit'


# ---------------------------------------------------------------- objective history

def test_plot_obj_func_plain(tmp_path, windpower_file):
    folder = tmp_path / 'weight_0.5'
    write_optimize_results(folder, len(GAS_STATES))
    figures = set_context(folder, make_kwens(GAS_STATES), make_kwsim(windpower_file))
    pr.plot_obj_func()
    assert (figures / 'obj_func.png').exists()


def test_plot_obj_func_epf(tmp_path, windpower_file):
    folder = tmp_path / 'weight_0.5'
    write_optimize_results(folder, len(HYDROGEN_STATES), num_free=2, epf=True)
    figures = set_context(folder, make_kwens(HYDROGEN_STATES), make_kwsim(windpower_file, True))
    pr.plot_obj_func(2.0)
    assert (figures / 'obj_func_epf.png').exists()


def test_collect_epf_history(tmp_path):
    folder = tmp_path / 'weight_0.5'
    write_optimize_results(folder, len(HYDROGEN_STATES), num_free=2, num_iter=2, epf=True)
    pr.set_plot_context(path_to_files_=str(folder) + os.sep)
    history = pr._collect_epf_history(os.listdir(folder))
    assert [h['outer_it'] for h in history] == [0, 1]
    assert [h['inner_it'] for h in history] == [[0, 1], [0, 1]]
    np.testing.assert_allclose(history[1]['penalty'], [0.5, 0.5])


def test_plot_obj_func_without_results_does_nothing(tmp_path, windpower_file):
    folder = tmp_path / 'empty'
    figures = set_context(folder, make_kwens(GAS_STATES), make_kwsim(windpower_file))
    pr.plot_obj_func()
    assert list(figures.iterdir()) == []


# ---------------------------------------------------------------- state plots

def test_plot_state_gas(tmp_path, windpower_file):
    folder = tmp_path / 'weight_0.5'
    write_optimize_results(folder, len(GAS_STATES))
    figures = set_context(folder, make_kwens(GAS_STATES), make_kwsim(windpower_file))
    pr.plot_state([1] * len(GAS_STATES), order='C')
    names = sorted(p.name for p in figures.iterdir())
    assert names == [f'variable_{i}.png' for i in range(len(GAS_STATES))]


def test_plot_state_hydrogen_with_free_variables(tmp_path, windpower_file):
    folder = tmp_path / 'weight_0.5'
    write_optimize_results(folder, len(HYDROGEN_STATES), num_free=2, epf=True)
    figures = set_context(folder, make_kwens(HYDROGEN_STATES), make_kwsim(windpower_file, True))
    pr.plot_state([1] * len(GAS_STATES), order='C')
    assert (figures / 'free_variable_0.png').exists()
    assert (figures / 'free_variable_1.png').exists()
    assert (figures / 'variable_6.png').exists()


# ---------------------------------------------------------------- pareto point

def test_plot_pareto_point_gas_histograms(tmp_path, windpower_file):
    folder = tmp_path / 'weight_0.5'
    npv, co2 = write_pareto_point(folder, ne=20)
    figures = set_context(folder, make_kwens(GAS_STATES), make_kwsim(windpower_file), show=True)
    pr.plot_pareto_point_gas()
    assert (figures / 'pareto_point.png').exists()

    ax_npv, ax_co2 = pr.plt.gcf().axes
    assert ax_npv.get_xlabel() == 'Million USD'
    assert ax_co2.get_xlabel() == 'Kilotonnes'
    # Panes are stacked vertically.
    assert ax_npv.get_position().y0 > ax_co2.get_position().y0
    mean_co2 = [t.get_text() for t in ax_co2.get_legend().get_texts()][1]
    assert mean_co2 == f'Mean: {np.mean(co2.sum(axis=1)) / 1e3:,.1f}'


def test_plot_pareto_point_gas_missing_file(tmp_path, windpower_file):
    folder = tmp_path / 'weight_0.5'
    figures = set_context(folder, make_kwens(GAS_STATES), make_kwsim(windpower_file))
    pr.plot_pareto_point_gas()
    assert not (figures / 'pareto_point.png').exists()


def test_plot_pareto_point_hydrogen(tmp_path, windpower_file):
    folder = tmp_path / 'weight_0.5'
    write_optimize_results(folder, len(HYDROGEN_STATES), num_free=2, epf=True)
    write_pareto_point(folder, co2_level=0.0)
    figures = set_context(folder, make_kwens(HYDROGEN_STATES), make_kwsim(windpower_file, True))
    pr.plot_pareto_point_hydrogen()
    assert (figures / 'pareto_point.png').exists()


def test_compute_total_capex_from_folder(tmp_path, windpower_file):
    folder = tmp_path / 'weight_0.5'
    write_optimize_results(folder, len(HYDROGEN_STATES), num_free=2, epf=True)
    kwens = make_kwens(HYDROGEN_STATES)
    kwsim = make_kwsim(windpower_file, True)
    pr.set_plot_context(kwens=kwens, kwsim=kwsim)

    x = np.load(folder / 'optimize_result_2_1.npz')['x']
    n_wt = x[-2] * 50
    n_h2 = x[-1] * 1000
    capex = pr._compute_total_capex_from_folder(str(folder), pr._config_dict())
    assert capex == pytest.approx(n_wt * 1.0e6 + n_h2 * 2.0e3)


def test_compute_total_capex_is_none_for_gas(tmp_path, windpower_file):
    folder = tmp_path / 'weight_0.5'
    write_optimize_results(folder, len(GAS_STATES))
    pr.set_plot_context(kwens=make_kwens(GAS_STATES), kwsim=make_kwsim(windpower_file))
    assert pr._compute_total_capex_from_folder(str(folder), pr._config_dict()) is None


# ---------------------------------------------------------------- pareto curve and wind

def test_plot_pareto_curve_gas_uses_co2(tmp_path, windpower_file):
    for i, weight in enumerate([0.0, 0.5, 1.0]):
        folder = tmp_path / f'weight_{weight}'
        write_optimize_results(folder, len(GAS_STATES))
        write_pareto_point(folder, npv_level=3e9 - i * 1e8, co2_level=200 - i * 20, seed=i)
    pr.set_plot_context(results_root_=tmp_path, kwens=make_kwens(GAS_STATES),
                        kwsim=make_kwsim(windpower_file), kwopt={}, show_figures=True)
    pr.plot_pareto_curve()

    assert (tmp_path / 'pareto_curve.png').exists()
    ax = pr.plt.gcf().axes[0]
    assert 'CO$_2$' in ax.get_xlabel()
    assert len(ax.collections[0].get_offsets()) == 3


def test_plot_pareto_curve_hydrogen_uses_capex(tmp_path, windpower_file):
    for i, weight in enumerate([0.5, 0.9]):
        folder = tmp_path / f'weight_{weight}'
        write_optimize_results(folder, len(HYDROGEN_STATES), num_free=2, epf=True, seed=i)
        write_pareto_point(folder, co2_level=0.0, seed=i)
    pr.set_plot_context(results_root_=tmp_path, kwens=make_kwens(HYDROGEN_STATES),
                        kwsim=make_kwsim(windpower_file, True), kwopt={}, show_figures=True)
    pr.plot_pareto_curve()
    ax = pr.plt.gcf().axes[0]
    assert ax.get_xlabel() == 'CAPEX (Million USD)'


def test_plot_pareto_curve_selected_weights(tmp_path, windpower_file):
    for weight in [0.0, 0.5, 1.0]:
        folder = tmp_path / f'weight_{weight}'
        write_optimize_results(folder, len(GAS_STATES))
        write_pareto_point(folder)
    pr.set_plot_context(results_root_=tmp_path, kwens=make_kwens(GAS_STATES),
                        kwsim=make_kwsim(windpower_file), kwopt={}, show_figures=True)
    pr.plot_pareto_curve(selected_weights=[0.0, 1.0])
    ax = pr.plt.gcf().axes[0]
    assert len(ax.collections[0].get_offsets()) == 2


def test_plot_pareto_curve_legacy_folder_names(tmp_path, windpower_file):
    for weight in [0.5, 0.9]:
        folder = tmp_path / f'run2w{weight}'
        write_optimize_results(folder, len(GAS_STATES))
        write_pareto_point(folder)
    pr.set_plot_context(results_root_=tmp_path, kwens=make_kwens(GAS_STATES),
                        kwsim=make_kwsim(windpower_file), kwopt={})
    pr.plot_pareto_curve()
    assert (tmp_path / 'pareto_curve_run2.png').exists()


def test_plot_wind_power_profiles(tmp_path, windpower_file):
    pr.set_plot_context(results_root_=tmp_path, kwsim=make_kwsim(windpower_file))
    pr.plot_wind_power_profiles()
    assert (tmp_path / 'wind_power_profiles.png').exists()
