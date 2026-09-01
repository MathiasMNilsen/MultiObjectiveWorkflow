import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nint = 9
folder = 'results_tol_1.0%_n1000'


def plot_interval(i):
    
    # Plot field data
    data_optimal = pd.read_csv(f'{folder}/Interval{i}/optimal/production.csv', index_col=0)
    data_target  = pd.read_csv(f'{folder}/Interval{i}/target/production.csv', index_col=0)
    data_proxy   = pd.read_csv(f'{folder}/Interval{i}/proxy/production.csv', index_col=0)

    # Make plot
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(13, 6))

    # FOPR
    # --------------------------------------------------------------
    ndays = data_optimal.index.size
    ax[0, 0].step(
        range(ndays), 
        data_optimal['FOPR'], 
        where='mid',
        label='Optimal',
        color='black',
        zorder=3,
    )
    ax[0, 0].step(
        range(ndays), 
        data_target['FOPR'], 
        where='mid',
        label='Target', 
        color='gray',
        zorder=1
    )
    ax[0, 0].step(
        range(ndays), 
        data_proxy['FOPR'], 
        where='mid',
        label='Proxy',
        color='tab:red',
        zorder=2
    ) 
    
    ax[0, 0].grid(color='lightgray', zorder=0)
    ax[0, 0].set_title('FOPR')
    ax[0, 0].legend(frameon=False)
    # --------------------------------------------------------------


    # FGPR
    # --------------------------------------------------------------
    ax[0, 1].step(
        range(ndays), 
        data_optimal['FGPR'], 
        where='mid',
        label='Optimal',
        color='black',
        zorder=3,
    )
    ax[0, 1].step(
        range(ndays), 
        data_target['FGPR'], 
        where='mid',
        label='Target', 
        color='gray',
        zorder=1
    )
    ax[0, 1].step(
        range(ndays), 
        data_proxy['FGPR'], 
        where='mid',
        label='Proxy',
        color='tab:red',
        zorder=2
    )   

    ax[0, 1].grid(color='lightgray', zorder=0)
    ax[0, 1].set_title('FGPR')
    # --------------------------------------------------------------


    # FWPR
    # --------------------------------------------------------------
    ax[1, 0].step(
        range(ndays), 
        data_optimal['FWPR'], 
        where='mid',
        label='Optimal',
        color='black',
        zorder=3,
    )
    ax[1, 0].step(
        range(ndays), 
        data_target['FWPR'], 
        where='mid',
        label='Target', 
        color='gray',
        zorder=1
    )
    ax[1, 0].step(
        range(ndays), 
        data_proxy['FWPR'], 
        where='mid',
        label='Proxy',
        color='tab:red',
        zorder=2
    )   

    ax[1, 0].grid(color='lightgray', zorder=0)
    ax[1, 0].set_title('FWPR')
    # --------------------------------------------------------------


    # FWIR
    # --------------------------------------------------------------
    ax[1, 1].step(
        range(ndays), 
        data_optimal['FWIR'], 
        where='mid',
        label='Optimal',
        color='black',
        zorder=3,
    )
    ax[1, 1].step(
        range(ndays), 
        data_target['FWIR'], 
        where='mid',
        label='Target', 
        color='gray',
        zorder=1
    )
    ax[1, 1].step(
        range(ndays), 
        data_proxy['FWIR'], 
        where='mid',
        label='Proxy',
        color='tab:red',
        zorder=2
    )
    ax[1, 1].grid(color='lightgray', zorder=0)
    ax[1, 1].set_title('FWIR')
    # --------------------------------------------------------------


    # Power Demand
    # --------------------------------------------------------------
    consumption_optimal = pd.read_csv(f'{folder}/Interval{i}/optimal/consumption.csv', index_col=0)
    consumption_target  = pd.read_csv(f'{folder}/Interval{i}/target/consumption.csv', index_col=0)
    consumption_proxy   = pd.read_csv(f'{folder}/Interval{i}/proxy/consumption.csv', index_col=0)

    ax[0, 2].step(
        range(ndays), 
        consumption_optimal['power_demand [MW]'], 
        where='mid',
        label='Optimal',
        color='black',
        zorder=4,
    )

    ax[0, 2].step(
        range(ndays), 
        consumption_optimal['windpower [MW]'], 
        where='mid',
        label='Wind Power', 
        color='tab:blue',
        zorder=3
    )

    ax[0, 2].step(
        range(ndays), 
        consumption_target['power_demand [MW]'],
        where='mid',
        label='Target',
        color='gray',
        zorder=1
    )

    ax[0, 2].step(
        range(ndays), 
        consumption_proxy['power_demand [MW]'],
        where='mid',
        label='Proxy',
        color='tab:red',
        zorder=2
    )
    ax[0, 2].grid(color='lightgray', zorder=0)
    ax[0, 2].set_title('Power Demand')
    # --------------------------------------------------------------


    # Emissions (found un consumption file)
    # --------------------------------------------------------------
    ax[1, 2].step(
        range(ndays), 
        consumption_optimal['emission_rate [ton/day]'], 
        where='mid',
        label='Optimal',
        color='black',
        zorder=3,
    )

    ax[1, 2].step(
        range(ndays), 
        consumption_target['emission_rate [ton/day]'], 
        where='mid',
        label='Target', 
        color='gray',
        zorder=1
    )

    ax[1, 2].step(
        range(ndays), 
        consumption_proxy['emission_rate [ton/day]'], 
        where='mid',
        label='Proxy',
        color='tab:red',
        zorder=2
    )   
    ax[1, 2].grid(color='lightgray', zorder=0)
    ax[1, 2].set_title('Emissions')
    # --------------------------------------------------------------



    plt.tight_layout()
    fig.savefig('temp_plot.png', dpi=300)




plot_interval(9)