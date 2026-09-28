import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get NPV
def plot_npv(savefolder):

    if isinstance(savefolder, str):
        savefolder = [savefolder]
    assert isinstance(savefolder, list)

    npv = {k.split('_')[1]: [] for k in savefolder}

    for f, folder in enumerate(savefolder):
        it = 0
        reading = True
        while reading:
            file = f'{folder}/optimize_result_{it}.npz'
            try:
                npv[list(npv.keys())[f]].append(-np.load(file)['fun'])
                it += 1
            except FileNotFoundError:
                reading = False

    # Make Plot
    fig, ax = plt.subplots(figsize=(5.5, 3))

    for i, (method, val) in enumerate(npv.items()):
        ax.plot(np.arange(len(val)), val, marker='o', linestyle='-', label=method)

        def f(x):
            return 100*(x-val[0])/val[0]
        secaxy = ax.secondary_yaxis('right', functions=(f, f))
        secaxy.tick_params(axis='y', colors='cadetblue')
        secaxy.set_ylabel('relative increase [%]', color='cadetblue')

    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(-0.5, max(len(npv) for npv in npv.values())+0.5)
    ax.set_xlabel('iteration')
    ax.set_ylabel('NPV [Billion $]')
    ax.legend()

    plt.tight_layout()
    plt.show()
    #fig.savefig(f'{folder}/results.png', dpi=300, bbox_inches='tight')


def plot_stateX(savefolder):
    stateX = pd.read_csv(f'{savefolder}/stateX.csv', index_col=None)
    keysX = stateX.columns.tolist()

    fig, ax = plt.subplots(ncols=2, figsize=(10, 3))
    print(ax)

    # Producers (5 first columns)
    for i, key in enumerate(keysX[:5]):
        ax[0].plot(stateX.index, stateX[key], marker='o', linestyle='-', label=key)
    ax[0].set_title('Producers')
    ax[0].grid(alpha=0.25)
    ax[0].legend(ncol=2)

    # Injectors (remaining columns)
    for i, key in enumerate(keysX[5:]):
        ax[1].plot(stateX.index, stateX[key], marker='o', linestyle='-', label=key)
    ax[1].set_title('Injectors')
    ax[1].grid(alpha=0.25)
    ax[1].legend()

    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    #plot_npv(savefolder=['results_NewtonCG', 'results_BFGS'])
    plot_stateX('results_NewtonCG')