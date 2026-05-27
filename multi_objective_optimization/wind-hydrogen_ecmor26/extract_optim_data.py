from resdata.summary import Summary
import pandas as pd
import datetime as dt
import numpy as np
from input_output import read_config

case = Summary('En_0/DROGON')

final_data = {}
kwopt, kwsim, kwen = read_config.read_yaml('config_pareto.yaml')
kwsim['reportpoint'][0] = dt.datetime(2020, 7, 2, 0, 0)
true_prim = [kwsim['reporttype'], kwsim['reportpoint']]
#dt_times = [case.start_time + dt.timedelta(days=el) for el in true_prim[1]]
all_data = case.pandas_frame(time_index=true_prim[1])
final_data['RATE_A1'] = all_data['WOPR:A1']
final_data['RATE_A2'] = all_data['WOPR:A2']
final_data['RATE_A3'] = all_data['WOPR:A3']
final_data['RATE_A4'] = all_data['WOPR:A4']
final_data['RATE_OP5'] = all_data['WOPR:OP5']
final_data['RATE_A5'] = all_data['WWIR:A5']
final_data['RATE_A6'] = all_data['WWIR:A6']
np.savez('../results_ecmor26/run2w0.5/final_data.npz', final_data=final_data)
