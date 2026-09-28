import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
import numpy as np
import os
import re
from typing import cast
import yaml
import pandas as pd

mode = 'publication'  # 'presentation' or 'publication'
if mode == 'presentation':
    plt.rcParams.update({
        'font.size': 18,              # base size (ticks)
        'axes.titlesize': 28,         # figure / axes titles
        'axes.labelsize': 22,         # x and y labels
        'xtick.labelsize': 18,        # x tick labels
        'ytick.labelsize': 18,        # y tick labels
        'legend.fontsize': 18,        # legend
    })
    figsize = 18.0
elif mode == 'publication':
    plt.rcParams.update({
        'font.size': 12,  # base text
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    figsize = 12.0
ratio = 16 / 9

def _load_state_names(path_to_config, num_state_vars):
    config_path = os.path.join(path_to_config, 'config.yaml')
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None

    try:
        state_names = cfg.get('ensemble', {}).get('state', None)
        if not isinstance(state_names, list):
            return None
        if len(state_names) < num_state_vars:
            state_names = state_names + [None] * (num_state_vars - len(state_names))
        else:
            state_names = state_names[:num_state_vars]
        return state_names
    except Exception:
        return None

def _load_state_limits(path_to_config, state_names):
    """
    Load (a, b) limits for each state from config.yaml:
    under `ensemble: prior_<state_name>: limits: [a, b]`.

    Returns a list of (a, b) tuples, same length as state_names.
    If limits are missing/invalid, returns (0.0, 1.0) as a fallback.
    """
    config_path = os.path.join(path_to_config, 'config.yaml')
    if not os.path.exists(config_path):
        return [(0.0, 1.0)] * len(state_names)

    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return [(0.0, 1.0)] * len(state_names)

    limits = []
    ens = cfg.get('ensemble', {})
    transform = ens.get('transform', True)
    if not transform:
        return limits, transform
    for name in state_names:
        if name is None:
            limits.append((0.0, 1.0))
            continue
        prior_key = f'prior_{name}'
        prior_cfg = ens.get(prior_key, {})
        lim = prior_cfg.get('limits', None)
        if isinstance(lim, (list, tuple)) and len(lim) == 2:
            try:
                a, b = float(lim[0]), float(lim[1])
                limits.append((a, b))
            except Exception:
                limits.append((0.0, 1.0))
        else:
            limits.append((0.0, 1.0))
    return limits, transform

def _load_time_index(path_to_config, num_steps):
    """
    Create num_steps+1 dates between start and end (inclusive),
    so last date matches end of last stair step.
    """
    config_path = os.path.join(path_to_config, 'config.yaml')
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None

    try:
        rp = cfg.get('fwdsim', {}).get('reportpoint', {})
        start = rp.get('start', None)
        end = rp.get('end', None)
        if start is None or end is None:
            return None

        time_index = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), periods=num_steps + 1)
        if len(time_index) != num_steps + 1:
            return None
        return time_index
    except Exception:
        return None

def _inverse_transform(normalized_values, a, b):
    """
    Inverse of mapping (a,b)->(0,1) assuming:
    normalized = (x - a)/(b - a)
    => x = a + normalized*(b - a)
    """
    scale = (b - a)
    if scale == 0:
        return np.full_like(normalized_values, a, dtype=float)
    return a + normalized_values * scale

def _parse_result_indices(filename):
    prefix = 'optimize_result_'
    suffix = '.npz'
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        return None

    parts = filename[len(prefix):-len(suffix)].split('_')
    try:
        if len(parts) == 1:
            return int(parts[0]), None
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return None

def _format_y_axis(ax, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    magnitude = np.nanmax(np.abs(values))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    if magnitude >= 1e4 or (0 < magnitude < 1e-2):
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        ax.ticklabel_format(style='plain', axis='y')

def _set_axis_padding(ax, values, anchor_zero=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if np.isclose(vmin, vmax):
        span = max(abs(vmax), 1.0)
        pad = 0.1 * span
    else:
        pad = 0.08 * (vmax - vmin)

    lower = 0.0 if anchor_zero and vmin >= 0 else vmin - pad
    upper = vmax + pad
    ax.set_ylim(lower, upper)

def _split_label_and_unit(label):
    """Split 'name (unit)' into ('name', 'unit')."""
    text = '' if label is None else str(label).strip()
    if text.endswith(')') and '(' in text:
        idx = text.rfind('(')
        name = text[:idx].strip()
        unit = text[idx + 1:-1].strip()
        if name and unit:
            return name, unit
    return text, ''

def _collect_epf_history(results, obj_scaling=None):
    history = {}
    for filename in sorted(results):
        parsed = _parse_result_indices(filename)
        if parsed is None or parsed[1] is None:
            continue

        outer_it, inner_it = parsed
        info = np.load(os.path.join(path_to_files, filename), allow_pickle=True)
        objective = np.array(info['fun'], dtype=float)
        penalty = np.array([], dtype=float)
        if 'epf' in info:
            epf_info = info['epf'][()]
            if isinstance(epf_info, dict) and 'penalty' in epf_info:
                penalty = np.array(epf_info['penalty'], dtype=float)

        obj_value = float(np.mean(objective))
        pen_value = float(np.mean(penalty)) if penalty.size else 0.0
        if obj_scaling is not None:
            obj_value *= -obj_scaling
            pen_value *= obj_scaling

        history.setdefault(outer_it, {})[inner_it] = {
            'objective': obj_value,
            'penalty': pen_value,
        }

    ordered_history = []
    for outer_it in sorted(history):
        inner_history = history[outer_it]
        ordered_inner = sorted(inner_history)
        ordered_history.append({
            'outer_it': outer_it,
            'inner_it': ordered_inner,
            'objective': np.array([inner_history[idx]['objective'] for idx in ordered_inner], dtype=float),
            'penalty': np.array([inner_history[idx]['penalty'] for idx in ordered_inner], dtype=float),
        })
    return ordered_history

def plot_obj_func(obj_scaling=None):
    """
    Plot the objective function vs. iterations.
    """
    files = os.listdir(path_to_files)
    results = sorted(name for name in files if "optimize_result" in name)
    if len(results) == 0:
        return
    obj = []
    first_result = _parse_result_indices(results[0])
    if first_result is not None and first_result[1] is not None:  # there is an epf outer loop index in the results
        history = _collect_epf_history(results, obj_scaling=obj_scaling)
        if not history:
            return

        fig, axes = plt.subplots(
            2, 1, figsize=(figsize, figsize/ratio), sharex=True,
            gridspec_kw={'height_ratios': [3, 1.5], 'hspace': 0.12}
        )
        ax_obj = cast(Axes, axes[0])
        ax_pen = cast(Axes, axes[1])
        cmap = plt.get_cmap('tab10')
        gap = 1.5
        x_cursor = 0.0
        x_tick_pos = []
        x_tick_labels = []
        all_obj = []
        all_pen = []

        for loop_idx, loop_data in enumerate(history):
            color = cmap(loop_idx % cmap.N)
            x_vals = np.arange(len(loop_data['inner_it']), dtype=float) + x_cursor
            if x_vals.size == 0:
                continue

            obj_vals = loop_data['objective']
            pen_vals = loop_data['penalty']
            all_obj.extend(obj_vals.tolist())
            all_pen.extend(pen_vals.tolist())

            span_start = x_vals[0] - 0.35
            span_end = x_vals[-1] + 0.35
            if loop_idx % 2 == 0:
                ax_obj.axvspan(span_start, span_end, color='0.96', zorder=0)
                ax_pen.axvspan(span_start, span_end, color='0.96', zorder=0)

            ax_obj.plot(
                x_vals, obj_vals, color=color, linewidth=2.6,
                marker='o', markersize=6, markeredgewidth=0,
                label=f'Outer loop {loop_data["outer_it"]}'
            )
            ax_pen.plot(
                x_vals, pen_vals, color=color, linewidth=2.0,
                linestyle='--', marker='s', markersize=5, markeredgewidth=0
            )
            ax_pen.fill_between(x_vals, 0.0, pen_vals, color=color, alpha=0.10)

            ax_obj.scatter(x_vals[-1], obj_vals[-1], s=90, facecolor='white', edgecolor=color, linewidth=2, zorder=4)
            ax_pen.scatter(x_vals[-1], pen_vals[-1], s=80, facecolor='white', edgecolor=color, linewidth=2, zorder=4)

            x_tick_pos.append(np.mean(x_vals))
            x_tick_labels.append(f'Outer {loop_data["outer_it"]}\n({len(x_vals)-1} it.)')
            x_cursor = x_vals[-1] + 1 + gap

            if loop_idx < len(history) - 1:
                separator = x_vals[-1] + (1 + gap) / 2
                ax_obj.axvline(separator, color='0.80', linewidth=1.2, linestyle=':')
                ax_pen.axvline(separator, color='0.80', linewidth=1.2, linestyle=':')

        all_x = np.concatenate([
            np.arange(len(loop_data['inner_it']), dtype=float) + offset
            for loop_data, offset in zip(
                history,
                np.cumsum([0.0] + [len(loop['inner_it']) + gap for loop in history[:-1]])
            )
        ])
        all_obj_arr = np.asarray(all_obj, dtype=float)
        all_pen_arr = np.asarray(all_pen, dtype=float)

        # Mark the best objective among points where penalty < 1% of the max penalty.
        max_pen = np.nanmax(all_pen_arr) if all_pen_arr.size > 0 else 0.0
        threshold = 0.01 * max_pen
        low_penalty_mask = all_pen_arr <= threshold
        if np.any(low_penalty_mask):
            candidate_idx = np.where(low_penalty_mask)[0]
            best_idx = candidate_idx[np.argmax(all_obj_arr[candidate_idx])]
        else:
            best_idx = int(np.argmax(all_obj_arr))

        best_x = all_x[best_idx]
        best_obj = all_obj_arr[best_idx]
        ax_obj.scatter(best_x, best_obj, s=170, marker='*', color='gold', edgecolor='black', linewidth=0.8, zorder=5)

        for ax in (ax_obj, ax_pen):
            ax.grid(True, axis='y', linestyle='--', alpha=0.35)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=16)

        ax_obj.set_ylabel('Objective value')
        ax_pen.set_ylabel('Penalty term')
        #ax_pen.set_xlabel('Inner iterations grouped by outer penalty loop')
        ax_pen.set_xticks(x_tick_pos)
        ax_pen.set_xticklabels(x_tick_labels)

        _format_y_axis(ax_obj, all_obj)
        _format_y_axis(ax_pen, all_pen)
        _set_axis_padding(ax_obj, all_obj)
        _set_axis_padding(ax_pen, all_pen, anchor_zero=True)

        fig.suptitle('Optimization progress across penalty loops', y=0.95)

        handles, labels = ax_obj.get_legend_handles_labels()
        #legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
        #                    ncol=min(3, len(history)), frameon=False)
        #for line in legend.legend_handles:
        #    if line is not None and hasattr(line, 'set_linewidth'):
        #        line.set_linewidth(3)

        #subtitle = 'Final point in each loop is highlighted. Gold star marks the best plotted objective.'
        #if obj_scaling is not None:
        #    subtitle += f' Objective scaling = {obj_scaling:.2g}.'
        #fig.text(0.5, 0.85, subtitle, ha='center', va='top', fontsize=18, color='0.35')

        fig.subplots_adjust(top=0.88, bottom=0.09, left=0.07, right=0.985, hspace=0.12)
        fig.savefig(str(path_to_figures) + '/obj_func_epf', bbox_inches='tight', pad_inches=0.04)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
        num_iter = len(results)
        for it in range(num_iter):
            info = np.load(str(path_to_files) + 'optimize_result_{}.npz'
                           .format(it), allow_pickle=True)
            val = info['fun']
            if obj_scaling is not None:
                val *= obj_scaling
            obj.append(val)
        obj = np.squeeze(np.array(obj))
        x_vals = np.arange(num_iter)
        if obj.ndim > 1:  # multiple models
            if np.min(obj.shape) == 1:
                ax.plot(x_vals, obj, '.', color='0.6', alpha=0.6)
            else:
                ax.plot(x_vals, obj, color='0.75', linestyle=':', linewidth=1.4)
            obj = np.mean(obj, axis=1)
        ax.plot(x_vals, obj, color='#1f77b4', marker='o', linewidth=2.8, markersize=7)
        ax.scatter(x_vals[-1], obj[-1], s=90, facecolor='white', edgecolor='#1f77b4', linewidth=2, zorder=4)
        ax.set_xticks(range(num_iter), minor=True)
        ax.tick_params(labelsize=16)
        ax.grid(True, axis='y', linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Iteration no.', size=20)
        ax.set_ylabel('Objective value', size=20)
        ax.set_title('Objective function', size=20)
        _format_y_axis(ax, obj)
        _set_axis_padding(ax, obj)
        plt.tight_layout()
        fig.savefig(str(path_to_figures) + '/obj_func')

def plot_state(num_var, order='F', state_actual=None):
    # load results
    files = os.listdir(path_to_files)
    results = [name for name in files if "optimize_result" in name]
    if not results:
        return

    ind = [i for i, ltr in enumerate(results[0]) if ltr == '_']
    if len(ind) > 2:
        inner_it = 0
        outer_it = 0
        for f in results:
            ind = [i for i, ltr in enumerate(f) if ltr == '_']
            ind.append(f.find('.npz'))
            outer_it = np.maximum(int(f[ind[1] + 1:ind[2]]), outer_it)
        for f in results:
            if f'optimize_result_{outer_it}' in f:
                ind = [i for i, ltr in enumerate(f) if ltr == '_']
                ind.append(f.find('.npz'))
                inner_it = np.maximum(int(f[ind[2] + 1:ind[3]]), inner_it)
        state_initial = np.load(str(path_to_files) + 'optimize_result_0_0.npz',
                                allow_pickle=True)['x']
        state_final = np.load(str(path_to_files) + f'optimize_result_{outer_it}_{inner_it}.npz',
                              allow_pickle=True)['x']
    else:
        num_iter = len(results) - 1
        state_initial = np.load(str(path_to_files) + 'optimize_result_0.npz',
                                allow_pickle=True)['x']
        state_final = np.load(str(path_to_files) + f'optimize_result_{num_iter}.npz',
                              allow_pickle=True)['x']

    # num\_var like \[5,2]
    if isinstance(num_var, int):
        num_var = [num_var]
    tot_var = sum(num_var)
    len_state = len(state_initial)
    num_steps = int(len_state / tot_var)

    # split time\-series and free variables (last entries)
    free_state_initial = state_initial[tot_var * num_steps:]
    free_state_final = state_final[tot_var * num_steps:]
    num_free = len(free_state_initial)

    state_initial = state_initial[:tot_var * num_steps]
    state_final = state_final[:tot_var * num_steps]
    state_initial = np.reshape(state_initial, (tot_var, num_steps), order=order)
    state_final = np.reshape(state_final, (tot_var, num_steps), order=order)

    # names and limits for *all* variables, incl. free ones
    total_named_vars = tot_var + num_free
    state_names = _load_state_names(path_to_config, total_named_vars)
    if state_names is None:
        state_names = [f'var_{i}' for i in range(total_named_vars)]
    limits, transform = _load_state_limits(path_to_config, state_names)

    # Only transfrom if transform is True in config, otherwise assume already in original scale
    if transform:
        # inverse transform time\-series part
        for idx in range(tot_var):
            a, b = limits[idx]
            state_initial[idx, :] = _inverse_transform(state_initial[idx, :], a, b)
            state_final[idx, :] = _inverse_transform(state_final[idx, :], a, b)

        # inverse transform free variables using last names (e.g. N\_WT, N\_H2)
        if num_free > 0:
            for i in range(num_free):
                a, b = limits[tot_var + i]
                free_state_initial[i] = _inverse_transform(
                    np.array([free_state_initial[i]]), a, b
                )[0]
                free_state_final[i] = _inverse_transform(
                    np.array([free_state_final[i]]), a, b
                )[0]

    # time index
    time_index = _load_time_index(path_to_config, num_steps)
    if time_index is not None:
        x_vals = time_index          # length num\_steps+1
        x_ticks = time_index         # all boundaries incl. last
        x_tick_labels = [t.strftime('%Y-%m') for t in time_index]
        x_label = 'Date'
    else:
        x_vals = np.arange(num_steps + 1)
        x_ticks = np.arange(num_steps + 1)
        x_tick_labels = x_ticks
        x_label = 'Index'

    # plot time\-dependent variables
    for i, k in enumerate(num_var):
        num = num_var[i] if len(num_var) >= i else num_var[0]
        c = int(np.ceil(np.sqrt(num)))
        r = int(np.ceil(num / c))
        f, ax = plt.subplots(r, c, figsize=(figsize/2, figsize/2/ratio), sharex=False)
        ax = np.array(ax).flatten()
        group_vals = []
        for w in np.arange(num):
            global_idx = sum(num_var[:i]) + w
            var_ini = state_initial[global_idx, :]
            var_fin = state_final[global_idx, :]
            group_vals.extend(var_ini.tolist())
            group_vals.extend(var_fin.tolist())

            # get the actural rates
            var_act = None
            if state_actual is not None:
                if state_names[global_idx] in state_actual.keys():
                    var_act = state_actual[state_names[global_idx]]
            
            if len(var_ini) > 1:
                ax[w].stairs(var_ini, x_vals, color='#4C78A8', baseline=None, linewidth=1.9, alpha=0.90)
                ax[w].stairs(var_fin, x_vals, color='#E45756', baseline=None, linewidth=2.2, alpha=0.95)
                if var_act is not None:
                    # Skip the first value so the original index serves as the N+1 edges
                    # for the remaining N values (stairs needs len(values)+1 edges).
                    act_values = var_act.values[1:]
                    act_edges = var_act.index
                    ax[w].stairs(act_values, act_edges, color='#2ca02c', baseline=None, linewidth=1.6, alpha=0.85, linestyle='--')
            else:
                ax[w].plot(x_vals[:-1], var_ini, 'o', color='#4C78A8', markersize=6)
                ax[w].plot(x_vals[:-1], var_fin, 'o', color='#E45756', markersize=6)

            ax[w].tick_params(direction='out', length=4, width=0.8)
            ax[w].set_xlabel(x_label if w >= num - c else '')

            if len(var_ini) > 1:
                if time_index is not None:
                    step = max(1, int(np.ceil(len(x_ticks) / 8)))
                    ax[w].set_xticks(x_ticks[::step])
                    ax[w].set_xticklabels(x_tick_labels[::step], rotation=30, ha='right')
                else:
                    ax[w].xaxis.set_major_locator(mticker.MaxNLocator(8, integer=True))

            name = state_names[global_idx]
            title = str(name) if name is not None else f'Variable {w + 1}'
            title_text, unit_text = _split_label_and_unit(f'{title} (Sm3/day)')
            ax[w].set_title(title_text, pad=6)
            ax[w].set_ylabel(unit_text)
            ax[w].grid(True, axis='y', linestyle='--', alpha=0.28)
            ax[w].spines['top'].set_visible(False)
            ax[w].spines['right'].set_visible(False)
            _format_y_axis(ax[w], np.concatenate([var_ini, var_fin]))
            _set_axis_padding(ax[w], np.concatenate([var_ini, var_fin]))

            if w == 0:
                line_ini = plt.Line2D([0], [0], color='#4C78A8', lw=2)
                line_fin = plt.Line2D([0], [0], color='#E45756', lw=2)
                legend_handles = [line_ini, line_fin]
                legend_labels = ['Initial', 'Final']
                if var_act is not None:
                    line_act = plt.Line2D([0], [0], color='#2ca02c', lw=1.6, linestyle='--')
                    legend_handles.append(line_act)
                    legend_labels.append('Actual')
                f.legend(legend_handles, legend_labels, loc='upper center',
                         ncol=len(legend_handles), frameon=False, bbox_to_anchor=(0.5, 0.995))

        # Use a shared y-range within each subplot group to improve comparability.
        if group_vals:
            group_vals = np.asarray(group_vals, dtype=float)
            group_vals = group_vals[np.isfinite(group_vals)]
            if group_vals.size > 0:
                gvmin, gvmax = np.min(group_vals), np.max(group_vals)
                gpad = 0.08 * (gvmax - gvmin) if not np.isclose(gvmin, gvmax) else 0.1 * max(abs(gvmax), 1.0)
                for w in np.arange(num):
                    ax[w].set_ylim(gvmin - gpad, gvmax + gpad)

        for idx in range(num, len(ax)):
            f.delaxes(ax[idx])

        f.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        f.savefig(str(path_to_figures) + f'/variable_{i}')

    # plot free variables (bars) with their own names and limits
    for i in range(num_free):
        f, ax = plt.subplots(1, 1, figsize=(figsize/2, figsize /2 / ratio))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        global_idx = tot_var + i
        name = state_names[global_idx] if global_idx < len(state_names) else None
        scaling = 1.0
        if isinstance(name, str) and 'WT' in name:
            scaling = 8.0 # MW per wind turbine
            yl = 'Rated wind capacity (MW)'
        elif isinstance(name, str) and 'H2' in name:
            yl = 'Hydrogen storage capacity (Tonne)'
        else:
            yl = 'Value'
        var_ini = free_state_initial[i] * scaling
        var_fin = free_state_final[i] * scaling

        # Fix y-axis range to configured prior limits for this free state.
        y_min = None
        y_max = None
        if (tot_var + i) < len(limits):
            a_lim, b_lim = limits[tot_var + i]
            y_min = min(a_lim, b_lim) * scaling
            y_max = max(a_lim, b_lim) * scaling
            if np.isclose(y_min, y_max):
                y_max = y_min + 1.0

        x_pos = np.array([0.0, 0.70])  # closer bar centers than default categorical spacing
        bars = ax.bar(x_pos, [var_ini, var_fin],
                      color=['#1f77b4', '#d62728'], width=0.52, edgecolor='black')
        max_abs_bar = max(abs(var_ini), abs(var_fin), 1.0)
        for bar, val in zip(bars, [var_ini, var_fin]):
            # Place annotation inside when there is room; otherwise place just above for readability.
            inside = abs(val) >= 0.18 * max_abs_bar
            if val >= 0:
                y_text = val - 0.09 * max_abs_bar if inside else val + 0.05 * max_abs_bar
                va = 'top' if inside else 'bottom'
            else:
                y_text = val + 0.09 * max_abs_bar if inside else val - 0.05 * max_abs_bar
                va = 'bottom' if inside else 'top'

            label = f'{val:.2f}'
            if isinstance(name, str) and 'WT' in name:
                n_turb = int(np.ceil(max(val, 0.0) / 8.0))
                label = f'{val:.2f}\n{n_turb} turbines'

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                label,
                ha='center',
                va=va,
                color='white' if inside else 'black',
                fontweight='bold',
                bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.40, 'pad': 1.6} if inside else None,
            )
        #ax.tick_params(labelsize=12)
        ax.set_xlabel('State')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Initial', 'Final'])
        global_idx = tot_var + i
        name = state_names[global_idx] if global_idx < len(state_names) else None
        title_text, unit_text = _split_label_and_unit(yl)
        ax.set_ylabel(unit_text)
        ax.set_title(title_text)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        else:
            _set_axis_padding(ax, [var_ini, var_fin], anchor_zero=True)

        f.tight_layout()
        f.savefig(str(path_to_figures) + f'/free_variable_{i}')

def plot_pareto_point():
    """
    Plot NPV distribution and CAPEX term breakdown.

    CAPEX terms are computed as:
      WT_term = N_WT * wt
      H2_term = N_H2 * h2
      Total   = WT_term + H2_term

    where wt/h2 are read from config constants.
    """
    data = np.load(path_to_files + 'pareto_point.npz', allow_pickle=True)
    npv_vals = np.asarray(data['npv'], dtype=float).ravel()
    if npv_vals.size == 0:
        return

    config_path = os.path.join(path_to_config, 'config.yaml')
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    state_names = cfg.get('ensemble', {}).get('state', [])
    if not isinstance(state_names, list):
        state_names = []

    # Identify free states from names (WT/H2) and recover their final values.
    free_pos = [i for i, name in enumerate(state_names)
                if isinstance(name, str) and ('WT' in name or 'H2' in name)]
    n_states = len(state_names)
    n_free = len(free_pos)
    wt_value = 0.0
    h2_value = 0.0

    # Load latest optimization result state vector.
    files = os.listdir(path_to_files)
    results = sorted(name for name in files if 'optimize_result' in name)
    state_final = None
    if results:
        parsed = []
        for name in results:
            idx = _parse_result_indices(name)
            if idx is not None:
                parsed.append((idx[0], -1 if idx[1] is None else idx[1], name))
        if parsed:
            _, _, latest_name = max(parsed)
            state_final = np.asarray(
                np.load(os.path.join(path_to_files, latest_name), allow_pickle=True)['x'], dtype=float
            )

    if state_final is not None and n_states > 0 and n_free > 0 and n_states > n_free:
        n_ts = n_states - n_free
        num_steps = int((state_final.size - n_free) / n_ts) if n_ts > 0 else 0
        if num_steps > 0 and (n_ts * num_steps + n_free) <= state_final.size:
            free_vals = np.array(state_final[n_ts * num_steps:n_ts * num_steps + n_free], dtype=float)

            # Inverse-transform free states when optimization is performed in normalized space.
            limits, transform = _load_state_limits(path_to_config, state_names)
            if transform and len(limits) == len(state_names):
                for j, pos in enumerate(free_pos):
                    a, b = limits[pos]
                    free_vals[j] = _inverse_transform(np.array([free_vals[j]]), a, b)[0]

            for j, pos in enumerate(free_pos):
                name = state_names[pos]
                if isinstance(name, str) and 'WT' in name:
                    wt_value = float(free_vals[j])
                elif isinstance(name, str) and 'H2' in name:
                    h2_value = float(free_vals[j])

    fwdsim = cfg.get('fwdsim', {}) if isinstance(cfg, dict) else {}
    npv_const = fwdsim.get('npv_const', {}) if isinstance(fwdsim, dict) else {}
    npt_const = fwdsim.get('npt_const', {}) if isinstance(fwdsim, dict) else {}
    wt_const = float(npt_const.get('wt', npv_const.get('wt', 0.0)))
    h2_const = float(npv_const.get('h2', npt_const.get('h2', 0.0)))

    wt_capex = wt_value * wt_const
    h2_capex = h2_value * h2_const
    total_capex = wt_capex + h2_capex

    fig, axes = plt.subplots(
        2, 1, figsize=(figsize, figsize / ratio),
        gridspec_kw={'height_ratios': [3.0, 2.0]}, constrained_layout=True
    )
    ax_hist = cast(Axes, axes[0])
    ax_capex = cast(Axes, axes[1])

    # Top: NPV histogram in million USD with summary markers.
    npv_musd = npv_vals / 1e6
    bins = int(np.clip(np.sqrt(npv_musd.size), 8, 30))
    counts, _, _ = ax_hist.hist(npv_musd, bins=bins, color='#1f77b4', alpha=0.80,
                                edgecolor='white', linewidth=0.8, label='Profit distribution')
    npv_mean = float(np.mean(npv_musd))
    npv_p10 = float(np.percentile(npv_musd, 10))
    npv_p90 = float(np.percentile(npv_musd, 90))
    ax_hist.axvline(npv_mean, color='#d62728', linewidth=2.2, label=f'Mean: {npv_mean:,.1f}')
    ax_hist.axvline(npv_p10, color='#ff7f0e', linewidth=1.8, linestyle='--', label=f'P10: {npv_p10:,.1f}')
    ax_hist.axvline(npv_p90, color='#2ca02c', linewidth=1.8, linestyle='--', label=f'P90: {npv_p90:,.1f}')
    ax_hist.set_title('Profit distribution')
    ax_hist.set_xlabel('Million USD')
    ax_hist.set_ylabel('Count')
    ax_hist.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)

    # Prefer legend inside when a top corner has reasonable free space.
    if counts.size > 0:
        k = max(1, int(np.ceil(0.25 * counts.size)))
        left_peak = float(np.max(counts[:k]))
        right_peak = float(np.max(counts[-k:]))
        overall_peak = float(np.max(counts))
        left_ratio = left_peak / (overall_peak + 1e-12)
        right_ratio = right_peak / (overall_peak + 1e-12)
        # If either corner is not too dense, place legend inside.
        corner_is_sparse = min(left_ratio, right_ratio) <= 0.80
    else:
        left_peak = 0.0
        right_peak = 0.0
        corner_is_sparse = True

    if corner_is_sparse:
        inside_loc = 'upper left' if left_peak <= right_peak else 'upper right'
        ax_hist.legend(loc=inside_loc, fontsize=12, frameon=True, framealpha=0.85)
    else:
        ax_hist.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
                       frameon=False, fontsize=12)

    # Bottom: WT/H2 CAPEX components and total.
    capex_terms = np.array([wt_capex, h2_capex, total_capex], dtype=float) / 1e6
    labels = ['WT term', 'H2 term', 'Total']
    colors = ['#4c78a8', '#f58518', '#54a24b']
    bars = ax_capex.bar(labels, capex_terms, color=colors, edgecolor='black', linewidth=0.8, width=0.62)
    ymax = float(np.max(capex_terms)) if capex_terms.size else 1.0
    label_pad = 0.08 * (ymax if ymax > 0 else 1.0)
    for bar, val in zip(bars, capex_terms):
        inside = val > 0.12 * max(ymax, 1e-12)
        y_text = val - label_pad if inside else val + 0.03 * max(ymax, 1.0)
        ax_capex.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text,
            f'{val:,.1f}',
            ha='center',
            va='top' if inside else 'bottom',
            color='white' if inside else 'black',
            fontweight='bold',
            bbox={'facecolor': 'black', 'edgecolor': 'none', 'alpha': 0.40, 'pad': 1.6} if inside else None,
        )
    ax_capex.set_title('CAPEX terms from final WT/H2 state')
    ax_capex.set_ylabel('Million USD')
    ax_capex.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax_capex.spines['top'].set_visible(False)
    ax_capex.spines['right'].set_visible(False)
    _set_axis_padding(ax_capex, capex_terms, anchor_zero=True)

    fig.savefig(str(path_to_figures) + '/pareto_point', bbox_inches='tight', pad_inches=0.04)

def _compute_total_capex_from_folder(results_folder, cfg):
    """Compute total CAPEX from the latest optimized free WT/H2 states in a folder."""
    state_names = cfg.get('ensemble', {}).get('state', []) if isinstance(cfg, dict) else []
    if not isinstance(state_names, list) or not state_names:
        return None

    files = os.listdir(results_folder)
    results = sorted(name for name in files if 'optimize_result' in name)
    parsed = []
    for name in results:
        idx = _parse_result_indices(name)
        if idx is not None:
            parsed.append((idx[0], -1 if idx[1] is None else idx[1], name))
    if not parsed:
        return None

    _, _, latest_name = max(parsed)
    state_final = np.asarray(
        np.load(os.path.join(results_folder, latest_name), allow_pickle=True)['x'], dtype=float
    )

    free_pos = [i for i, name in enumerate(state_names)
                if isinstance(name, str) and ('WT' in name or 'H2' in name)]
    n_states = len(state_names)
    n_free = len(free_pos)
    if n_free == 0 or n_states <= n_free:
        return None

    n_ts = n_states - n_free
    num_steps = int((state_final.size - n_free) / n_ts) if n_ts > 0 else 0
    if num_steps <= 0 or (n_ts * num_steps + n_free) > state_final.size:
        return None

    free_vals = np.array(state_final[n_ts * num_steps:n_ts * num_steps + n_free], dtype=float)
    limits, transform = _load_state_limits(path_to_config, state_names)
    if transform and len(limits) == len(state_names):
        for j, pos in enumerate(free_pos):
            a, b = limits[pos]
            free_vals[j] = _inverse_transform(np.array([free_vals[j]]), a, b)[0]

    wt_value = 0.0
    h2_value = 0.0
    for j, pos in enumerate(free_pos):
        name = state_names[pos]
        if isinstance(name, str) and 'WT' in name:
            wt_value = float(free_vals[j])
        elif isinstance(name, str) and 'H2' in name:
            h2_value = float(free_vals[j])

    fwdsim = cfg.get('fwdsim', {}) if isinstance(cfg, dict) else {}
    npv_const = fwdsim.get('npv_const', {}) if isinstance(fwdsim, dict) else {}
    npt_const = fwdsim.get('npt_const', {}) if isinstance(fwdsim, dict) else {}
    wt_const = float(npt_const.get('wt', npv_const.get('wt', 0.0)))
    h2_const = float(npv_const.get('h2', npt_const.get('h2', 0.0)))

    wt_capex = wt_value * wt_const
    h2_capex = h2_value * h2_const
    return wt_capex + h2_capex

def _collapse_duplicate_x(x_vals, y_vals):
    x_vals = np.asarray(x_vals, dtype=float).ravel()
    y_vals = np.asarray(y_vals, dtype=float).ravel()
    order = np.argsort(x_vals)
    x_vals = x_vals[order]
    y_vals = y_vals[order]

    unique_x, inverse = np.unique(x_vals, return_inverse=True)
    if unique_x.size == x_vals.size:
        return unique_x, y_vals

    y_collapsed = np.zeros_like(unique_x, dtype=float)
    counts = np.zeros_like(unique_x, dtype=float)
    for idx, inv in enumerate(inverse):
        y_collapsed[inv] += y_vals[idx]
        counts[inv] += 1.0
    y_collapsed /= np.maximum(counts, 1.0)
    return unique_x, y_collapsed

def _least_squares_curve_eval(x_vals, y_vals, num=250):
    """Return dense x/y samples from a least-squares polynomial approximation."""
    x_vals, y_vals = _collapse_duplicate_x(x_vals, y_vals)
    if x_vals.size < 2:
        return x_vals, y_vals

    dense_x = np.linspace(x_vals[0], x_vals[-1], num=max(num, 2))
    deg = min(2, x_vals.size - 1)
    try:
        coeffs = np.polyfit(x_vals, y_vals, deg=deg)
        dense_y = np.polyval(coeffs, dense_x)
    except Exception:
        dense_y = np.interp(dense_x, x_vals, y_vals)
    return dense_x, dense_y

def plot_pareto_curve(selected_weights=None):
    """Plot Pareto points across weights from ./results/<case>w<weight>/ folders.
    
    Parameters:
    -----------
    selected_weights : list of float, optional
        If provided, only plot results for these specific weights (e.g., [0.5, 0.9, 0.99]).
        If None, plot all available weights.
    """
    config_path = os.path.join(path_to_config, 'config.yaml')
    if not os.path.exists(config_path):
        return
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    cfg_dir = os.path.basename(os.path.normpath(path_to_config))
    match = re.search(r'run\d+', cfg_dir)
    if match is None:
        return
    case_name = match.group(0)

    if not os.path.isdir(results_root):
        return

    # Convert selected_weights to a set for fast lookup and handle tolerance for floating point comparison
    if selected_weights is not None:
        selected_weights = [float(w) for w in selected_weights]
        weights_set = set(selected_weights)
    else:
        weights_set = None

    prefix = f'{case_name}w'
    candidates = []
    for folder in os.listdir(results_root):
        folder_path = os.path.join(results_root, folder)
        if not os.path.isdir(folder_path) or not folder.startswith(prefix):
            continue
        weight_str = folder[len(prefix):]
        try:
            weight = float(weight_str)
        except ValueError:
            continue

        # Skip if weights are selected and this weight is not in the list (with tolerance)
        if weights_set is not None:
            weight_match = False
            for selected_w in selected_weights:
                if np.isclose(weight, selected_w, rtol=1e-9):
                    weight_match = True
                    break
            if not weight_match:
                continue

        pareto_file = os.path.join(folder_path, 'pareto_point.npz')
        if not os.path.exists(pareto_file):
            continue

        try:
            pareto_data = np.load(pareto_file, allow_pickle=True)
            npv = np.asarray(pareto_data['npv'], dtype=float).ravel()
            if npv.size == 0:
                continue
            npv_value = float(np.mean(npv))
            capex_value = _compute_total_capex_from_folder(folder_path, cfg)
            if capex_value is None:
                continue
        except Exception:
            continue

        candidates.append((weight, capex_value, npv_value))

    if not candidates:
        return

    candidates.sort(key=lambda x: x[0])
    weights = np.array([row[0] for row in candidates], dtype=float)
    capex_vals = np.array([row[1] for row in candidates], dtype=float) / 1e6
    npv_vals = np.array([row[2] for row in candidates], dtype=float) / 1e6

    fig, ax = plt.subplots(1, 1, figsize=(figsize * 0.75, figsize * 0.75 / ratio))
    ax.scatter(capex_vals, npv_vals, s=65, color='#1f77b4', zorder=4)

    curve_x, curve_y = _least_squares_curve_eval(capex_vals, npv_vals, num=400)
    if curve_x.size >= 2:
        ax.plot(curve_x, curve_y, color='orange', linewidth=2.2, alpha=0.9)

    x_span = max(np.ptp(capex_vals), 1.0)
    y_span = max(np.ptp(npv_vals), 1.0)
    for x, y, w in zip(capex_vals, npv_vals, weights):
        ax.text(x + 0.025 * x_span, y + 0.015 * y_span, f'w={w:g}', fontsize=10, color='0.25')

    ax.set_title('Pareto curve across weights')
    ax.set_xlabel('CAPEX (Million USD)')
    ax.set_ylabel('Profit (Million USD)')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Include the curve values when computing axis limits so the curve stays inside.
    all_y = np.concatenate([npv_vals, curve_y]) if curve_x.size >= 2 else npv_vals
    all_x = capex_vals  # curve x is within the scatter x range by construction
    x_pad = 0.08 * max(np.ptp(all_x), 1.0)
    y_pad = 0.08 * max(np.ptp(all_y), 1.0)
    ax.set_xlim(np.min(all_x) - x_pad, np.max(all_x) + x_pad)
    ax.set_ylim(np.min(all_y) - y_pad, np.max(all_y) + y_pad)

    os.makedirs(results_root, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_root, f'pareto_curve_{case_name}.png'), bbox_inches='tight', pad_inches=0.04)

def plot_wind_power_profiles():
    wind_power_ens = np.load('./init/wp_ens.npy')
    wind_power_ens = np.asarray(wind_power_ens, dtype=float)

    if wind_power_ens.ndim != 2 or wind_power_ens.shape[0] < 2:
        return

    rows_to_plot = [0, 1]
    num_days = wind_power_ens.shape[1]
    day_idx = np.arange(1, num_days + 1)

    fig, axes = plt.subplots(2, 1, figsize=(figsize, figsize / ratio), sharey=True)
    axes = np.atleast_1d(axes)

    smooth_window = 14
    kernel = np.ones(smooth_window, dtype=float) / smooth_window

    for ax, row_idx in zip(axes, rows_to_plot):
        profile = wind_power_ens[row_idx, :]
        smooth_profile = np.convolve(profile, kernel, mode='same')

        # Keep full daily variability but de-emphasize it to avoid visual clutter.
        ax.plot(day_idx, profile, color='#1f77b4', linewidth=0.8, alpha=0.30, label='Daily values')
        ax.plot(day_idx, smooth_profile, color='#d62728', linewidth=2.0, label=f'{smooth_window}-day mean')

        ax.set_xlabel('Day')
        ax.set_ylabel('Wind power (MW)')
        ax.set_title(f'Sample {row_idx+1}')
        ax.set_xlim(1, num_days)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(8, integer=True))
        ax.grid(True, linestyle='--', alpha=0.30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='0.35',
        framealpha=0.95,
        fancybox=True,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(results_root, 'wind_power_profiles'), bbox_inches='tight', pad_inches=0.04)


# Set paths
path_to_files = 'results_ecmor26/run2w0.5/'  # Load results from here
path_to_config = 'wind-hydrogen_ecmor26/'  # Extract config name from path
path_to_figures = path_to_files  # Save here
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)
results_root = './results'

# Load reference values (for scaling) and actual state
file = np.load('./init/ref_values.npz', allow_pickle=True)
f1_ref = file['co2'].sum(axis=1).mean()
f2_ref = file['npv'].mean()
final_data = np.load(path_to_files + 'final_data.npz', allow_pickle=True)
final_data = final_data['final_data'][()]
#print(final_data['RATE_A1'])

# Plot stuff
#plot_obj_func(2845473526.0)
#plot_state([1,1,1,1,1,1,1], order='C', state_actual=final_data)
#plot_pareto_point()
plot_pareto_curve(selected_weights=[0.5, 0.9, 0.99])
#plot_wind_power_profiles()

plt.show()
