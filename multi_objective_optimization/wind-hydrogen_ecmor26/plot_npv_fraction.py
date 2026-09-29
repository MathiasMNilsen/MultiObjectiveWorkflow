# plot npv fraction pie
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Colorblind friendly cycle
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

with open('./npv_data.pkl', 'rb') as f:
    data = pickle.load(f)
obj_scaling = 2845473526.0

labels = ['Oil production', 'Gas production', 'Water production', 'Water injection', 'CAPEX']
fractions = np.asarray(data, dtype=float) / obj_scaling

def _pct_label(pct):
    # Keep the chart readable by suppressing very small percentage labels.
    return f'{pct:.1f}%' if pct >= 1.0 else ''

f, ax = plt.subplots(1, 1, figsize=(6.4, 5.6), constrained_layout=True)
wedges, label_texts, pct_texts = ax.pie(
    fractions,
    labels=labels,
    colors=CB_color_cycle[0:5],
    startangle=90,
    counterclock=False,
    autopct=_pct_label,
    pctdistance=0.70,
    labeldistance=1.03,
    wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 1.2},
    textprops={'fontsize': 14, 'color': '#1a1a1a'}
)

for pct_text in pct_texts:
    pct_text.set_fontsize(12)
    pct_text.set_weight('bold')
    pct_text.set_color('white')

ax.text(0, 0, 'Typical NPV\n composition', ha='center', va='center', fontsize=14, color='#4a4a4a')
ax.set_aspect('equal')
figure_dir = '../results_ecmor26/'
f.savefig(figure_dir + 'npv_fraction_pie.svg', format='svg', facecolor='none', edgecolor='none',
          bbox_inches='tight', pad_inches=0.05)
f.savefig(figure_dir + 'npv_fraction_pie.pdf', format='pdf', facecolor='none', edgecolor='none',
          bbox_inches='tight', pad_inches=0.05)
plt.show()