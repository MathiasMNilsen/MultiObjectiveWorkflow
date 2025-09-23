import matplotlib.pyplot as plt
import numpy as np

ne = 50


def plot_pareto():

    weight = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    # colors for mean (dark version of ensemble)
    colors_mean = ['darkblue', 'darkorange', 'darkgreen', 'darkred', 'purple']

    fig, ax = plt.subplots(figsize=(6.5, 3.75))

    npv_wind = []
    co2_wind = []       

    npv_wind_mean = []
    co2_wind_mean = []

    for w, we in enumerate(weight):
       
        file_wind = np.load(f'results/weight{we}/pareto_point.npz')

        en_npv_wind = file_wind['npv']/1e9
        en_co2_wind = file_wind['co2'].sum(axis=1)/1000

        for n in range(ne):
            # ensemble points
            ax.scatter(en_co2_wind[n], en_npv_wind[n], color=colors[w], s=5, alpha=0.7)

        # mean points
        ax.scatter(np.mean(en_co2_wind), np.mean(en_npv_wind), c=colors[w], s=40, edgecolors='black', label=rf'$\omega={we}$')
    
        # append mean data
        npv_wind_mean.append(np.mean(en_npv_wind))
        co2_wind_mean.append(np.mean(en_co2_wind))
    

    # polynomial fit for wind
    z_wind = np.polyfit(co2_wind_mean, npv_wind_mean, 3)
    p_wind = np.poly1d(z_wind)
    x_wind = np.linspace(co2_wind_mean[0], co2_wind_mean[-1], 100)
    y_wind = p_wind(x_wind)
    ax.plot(x_wind, y_wind, color='gray', linewidth=1.25, ls='--', zorder=0, label=r'3$^{rd}$ deg. polyfit')


    # Plot refe3rence point
    file_ref = np.load('init/ref_values.npz', allow_pickle=True)
    f1_ref = file_ref['co2'].sum(axis=1).mean()/1000
    f2_ref = file_ref['npv'].mean()/1e9
    ax.scatter(f1_ref, f2_ref, color='gray', s=40, marker='x', label='initial', zorder=5)


    # Secondary x-axis
    def func1(x):
        return 100*(x-co2_wind_mean[0])/co2_wind_mean[0]
    secaxx = ax.secondary_xaxis('top', functions=(func1, func1))
    secaxx.set_xticks([-50, -40, -30, -20, -10, -0], ['-50%', '-40%', '-30%', '-20%', '-10%', '0%'], fontsize=7.5, color='cadetblue')
    #secaxx.set_xlabel(r'reduction from $\omega=0$ with only gas', fontsize=7.5, color='cadetblue')


    def func2(x):
        return 100*(x-npv_wind_mean[0])/npv_wind_mean[0]
    secaxy = ax.secondary_yaxis('right', functions=(func2, func2))
    secaxy.set_yticks([-50, -40, -30, -20, -10, -0], ['-50%', '-40%', '-30%', '-20%', '-10%', '0%'], fontsize=7.5, color='cadetblue')


    ax.spines['top'].set_color('cadetblue')
    ax.spines['right'].set_color('cadetblue')
    secaxx.tick_params(axis='x', colors='cadetblue')
    secaxy.tick_params(axis='y', colors='cadetblue')

    ax.legend(frameon=False, fontsize=8)
    ax.grid(color='gray', alpha=0.25)
    ax.set_xlabel(r'CO$_2$ emissions [kilotonnes]')
    ax.set_ylabel('NPV [billion USD]')

    plt.tight_layout()
    fig.savefig('pareto_front.png', dpi=300)

plot_pareto()