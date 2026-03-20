import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import yaml
import pandas as pd

mode = 'presentation'  # 'presentation' or 'publication'
if mode == 'presentation':
    plt.rcParams.update({
        'font.size': 18,              # base size (ticks)
        'axes.titlesize': 28,         # figure / axes titles
        'axes.labelsize': 22,         # x and y labels
        'xtick.labelsize': 18,        # x tick labels
        'ytick.labelsize': 18,        # y tick labels
        'legend.fontsize': 18,        # legend
    })
elif mode == 'publication':
    plt.rcParams.update({
        'font.size': 12,  # base text
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

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
        if not (start and end):
            return None

        time_index = pd.date_range(start=start, end=end, periods=num_steps + 1)
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

def plot_obj_func(obj_scaling=None):
    """
    Plot the objective function vs. iterations.
    """
    files = os.listdir(path_to_files)
    results = [name for name in files if "optimize_result" in name]
    if len(results) == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax2 = ax.twinx()
    obj = []
    pen = []
    ind = [i for i, ltr in enumerate(results[0]) if ltr == '_']
    if len(ind) > 2:  # there is an epf outer loop index in the results
        for f in results:
            ind = [i for i, ltr in enumerate(f) if ltr == '_']
            ind.append(f.find('.npz'))
            outer_it = int(f[ind[1]+1:ind[2]])
            inner_it = int(f[ind[2]+1:ind[3]])
            info = np.load(str(path_to_files) + 'optimize_result_{0}_{1}.npz'
                           .format(f[ind[1]+1:ind[2]], f[ind[2]+1:ind[3]]), allow_pickle=True)
            if not len(obj) > outer_it:
                obj.extend([] for _ in range(outer_it-len(obj)+1))
                pen.extend([] for _ in range(outer_it - len(pen) + 1))
            if not len(obj[outer_it]) > inner_it:
                obj[outer_it].extend([] for _ in range(inner_it-len(obj[outer_it])+1))
                pen[outer_it].extend([] for _ in range(inner_it - len(pen[outer_it]) + 1))
            val = np.array(info['fun'])
            penalty = info['epf'][()]['penalty'] if 'epf' in info and 'penalty' in info['epf'][()] else []
            if obj_scaling is not None:
                val *= -obj_scaling
                penalty *= obj_scaling
#            val += np.mean(penalty)
            obj[outer_it][inner_it] = np.mean(val)
            pen[outer_it][inner_it] = np.mean(penalty)
        handles = []
        labels = []
        for i in range(len(obj)):
            if len(obj[i]) > 1:
                line_obj, = ax.plot(obj[i], linewidth=2)
                handles.append(line_obj)
                labels.append('epf iter ' + str(i))
                if len(pen[i]):
                    line_pen, = ax2.plot(pen[i], linewidth=2, linestyle='--', color=line_obj.get_color())
                    handles.append(line_pen)
                    labels.append('pen iter ' + str(i))
            else:
                line_obj, = ax.plot(obj[i], 'd', markersize=10)
                handles.append(line_obj)
                labels.append('epf iter ' + str(i))
                if len(pen[i]):
                    line_pen, = ax2.plot(pen[i], 's', markersize=10, color=line_obj.get_color())
                    handles.append(line_pen)
                    labels.append('pen iter ' + str(i))
        ax2.set_ylabel('Penalty', size=20)
        ax2.tick_params(labelsize=16)
        ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax2.get_yaxis().get_offset_text().set_visible(True)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('Iteration no.', size=20)
        ax.set_ylabel('Objective Function', size=20)
        if any(any(pen[i]) for i in range(len(pen))):
            title = 'Objective function and penalty'
        else:
            title = 'Objective function'
        if obj_scaling is not None:
            title += f', scaling={obj_scaling:.2g}'
        ax.set_title(title, size=20)
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.85, 0.5), fontsize=16)
        plt.tight_layout()
        fig.savefig(str(path_to_figures) + '/obj_func_epf')
    else:
        num_iter = len(results)
        for it in range(num_iter):
            info = np.load(str(path_to_files) + 'optimize_result_{}.npz'
                           .format(it), allow_pickle=True)
            val = info['fun']
            if obj_scaling is not None:
                val *= obj_scaling
            obj.append(val)
        obj = np.squeeze(np.array(obj))
        if obj.ndim > 1:  # multiple models
            if np.min(obj.shape) == 1:
                ax.plot(obj, '.b')
            else:
                ax.plot(obj, 'b:')
            obj = np.mean(obj, axis=1)
        ax.plot(obj, 'rs-', linewidth=4, markersize=10)
        ax.set_xticks(range(num_iter), minor=True)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('Iteration no.', size=20)
        ax.set_ylabel('Value', size=20)
        ax.set_title('Objective function', size=20)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.get_yaxis().get_offset_text().set_visible(False)
        plt.tight_layout()
        fig.savefig(str(path_to_figures) + '/obj_func')

def plot_state(num_var, order='F'):
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
    ratio = 16 / 9
    figsize = 20
    for i, k in enumerate(num_var):
        num = num_var[i] if len(num_var) >= i else num_var[0]
        c = int(np.ceil(np.sqrt(num)))
        r = int(np.ceil(num / c))
        f, ax = plt.subplots(r, c, figsize=(figsize, figsize/ratio))
        ax = np.array(ax).flatten()
        for w in np.arange(num):
            global_idx = sum(num_var[:i]) + w
            var_ini = state_initial[global_idx, :]
            var_fin = state_final[global_idx, :]

            if len(var_ini) > 1:
                ax[w].stairs(var_ini, x_vals, color='blue', baseline=None, linewidth=2)
                ax[w].stairs(var_fin, x_vals, color='red', baseline=None, linewidth=2)
            else:
                ax[w].plot(x_vals[:-1], var_ini, 'sb')
                ax[w].plot(x_vals[:-1], var_fin, 'sr')

            ax[w].tick_params()
            ax[w].set_xlabel(x_label)
            ax[w].set_ylabel('Sm3/day')

            if len(var_ini) > 1:
                ax[w].set_xticks(x_ticks)
                ax[w].set_xticklabels(x_tick_labels, rotation=45, ha='right')

            name = state_names[global_idx]
            title = str(name) if name is not None else f'Variable {w + 1}'
            ax[w].set_title(title)

            if w == 0:
                ax[w].legend(['Initial', 'Final'])

        for idx in range(num, len(ax)):
            f.delaxes(ax[idx])

        f.tight_layout()
        f.savefig(str(path_to_figures) + f'/variable_{i}')

    # plot free variables (bars) with their own names and limits
    figsize = 10
    for i in range(num_free):
        f, ax = plt.subplots(1, 1, figsize=(figsize, figsize / ratio))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        global_idx = tot_var + i
        name = state_names[global_idx] if global_idx < len(state_names) else None
        scaling = 1.0
        if 'WT' in name:
            scaling = 8.0 # MW per wind turbine
            yl = 'Rated wind capacity (MW)'
        elif 'H2' in name:
            yl = 'Hydrogen storage capacity (tonne)'
        else:
            yl = 'Value'
        var_ini = free_state_initial[i] * scaling
        var_fin = free_state_final[i] * scaling
        ax.bar(['Initial', 'Final'], [var_ini, var_fin],
               color=['#1f77b4', '#d62728'], width=0.4, edgecolor='black')
        for idx, val in enumerate([var_ini, var_fin]):
            ax.text(idx, val, f'{val:.2f}', ha='center', va='bottom')
        #ax.tick_params(labelsize=12)
        ax.set_xlabel('State')
        global_idx = tot_var + i
        name = state_names[global_idx] if global_idx < len(state_names) else None
        ax.set_ylabel(yl)
        title = str(name) if name is not None else f'Free Variable {i + 1}'
        ax.set_title(title)

        f.tight_layout()
        f.savefig(str(path_to_figures) + f'/free_variable_{i}')

def plot_pareto_point():
    """
    Plot the Pareto front.
    """
    f = np.load(path_to_files + 'pareto_point.npz')
    f1 = f['capex']
    f2 = f['npv']
    plt.figure(figsize=(10, 6))
    plt.plot(f1 * np.ones(len(f2)), f2, 'o', color='blue', label='Pareto points')
    plt.xlabel('capex')
    plt.ylabel('npv')
    plt.legend()


# Set paths
path_to_files = '../results/wind-hydrogen_weight_0.5/'  # Load results from here
path_to_config = './'  # Extract config name from path
path_to_figures = './'  # Save here
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)

# Load reference values (for scaling)
file = np.load('../init/ref_values.npz', allow_pickle=True)
f1_ref = file['co2'].sum(axis=1).mean()
f2_ref = file['npv'].mean()
plot_obj_func(2845473526.0)
plot_state([5, 2])
plot_pareto_point()
plt.show()
