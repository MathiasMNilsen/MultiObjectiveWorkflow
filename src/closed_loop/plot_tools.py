import matplotlib.pyplot as plt

prod_wells = ['A1', 'A2', 'A3', 'A4', 'OP5']
inj_wells  = ['A5', 'A6']
prod_alpha = ['WOPR', 'WGPR', 'WWPR']
inj_alpha  = ['WWIR', 'WTHP']

axpos = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)]

def plot_interval(data, controls):
    fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(12, 8), sharex=True)

    time = data.index
    for w, well in enumerate(prod_wells):

        # Plot control
        y = controls[f'Rate{well}']
        ax[axpos[w]].plot(time, y, label=f'Control', color='black')
        
        # Plot production rates
        for a, alpha in enumerate(prod_alpha):
            if alpha == 'WGPR':
                y = data[f'{alpha}:{well}']/100
            else:
                y = data[f'{alpha}:{well}']

            ax[axpos[w]].plot(time, y, label=f'{alpha}')
            ax[axpos[w]].set_xlim(time[0], time[-1])
            ax[axpos[w]].set_title(f'Well {well}')
            ax[axpos[w]].set_ylabel('Rate')
            ax[axpos[w]].legend(frameon=False)
        

    plt.tight_layout()
    plt.show()