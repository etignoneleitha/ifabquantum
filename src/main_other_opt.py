import sys
sys.settrace
import numpy as np
import matplotlib.pyplot as plt
from utils.qaoa_qutip import *
from utils.gaussian_process import *
from utils.create_graphs import (create_random_graph,
                                 create_chair_graph,
                                 create_random_regular_graph,
                                 create_chain
                                 )
from utils.parameters import parse_command_line
from utils.default_params import *
import time
from pathlib import Path
import os
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
np.set_printoptions(precision=4, suppress=True)


    
################### PARAMETERS   ################

args = parse_command_line()

seed = args.seed
fraction_warmup = args.fraction_warmup
depth = args.p
trials = args.trials
i_trial = args.i_trial
num_nodes = args.num_nodes
nwarmup = args.nwarmup
average_connectivity = args.average_connectivity
problem = args.problem
nbayes = args.nbayes
optimizer_method = args.optimizer

param_range = np.array([[0.01, np.pi], [0.01, np.pi]])   # extremes where to search for the values of gamma and beta

global_time = time.time()

np.random.seed(seed)
num_graph = seed
name_plot = str(seed)


################ CREATE GRAPH AND QAOA ################


if problem == 'H2' or problem == 'H2_BK':
    G = create_chain(4)
elif problem == 'H2_reduced' or problem == 'H2_BK_reduced':
    G = create_chain(2)
else:
    G = create_random_regular_graph(num_nodes, degree=3, seed=1)
    
qaoa = qaoa_qutip(G, problem=problem)
gs_energy, gs_state, degeneracy = qaoa.gs_en, qaoa.gs_states, qaoa.deg

print('Information on the hamiltonian')
print('GS energy: ',gs_energy)
print('GS binary:', qaoa.gs_binary)
print('GS degeneracy: ', degeneracy)
print('GS :', qaoa.gs_states[0])
print('ham :, ', qaoa.H_c)
DEFAULT_PARAMS["seed"] = seed + i_trial


############### CREATE DATA FILE ###############

def angle_names_string():
    gamma_names = [f'GAMMA_{i}' for i in range(depth)]
    beta_names = [f'BETA_{i}' for i in range(depth)]
    
    angle_names = []
    for i in range(depth):
        angle_names.append(beta_names[i])
        angle_names.append(gamma_names[i])
        
    return angle_names
        
output_folder = Path(__file__).parents[1] / "output"
file_name = f'{optimizer_method}_p_{depth}_train_{nbayes}_num_nodes_{num_nodes}.dat'
data_ = []
angle_names = angle_names_string()
results_data_names = ['iter '] + angle_names +\
                                 [
                                 'energy',
                                 'best_energy',
                                 'approx_ratio',
                                 'best_approx_ratio',
                                 'fidelity',
                                 'best_fidelity',
                                 'variance',
                                 ]
                     
folder_name = file_name.split('.')[0]
folder = os.path.join(output_folder, folder_name)
os.makedirs(folder, exist_ok = True)
data_header = " ".join(["{:>7} ".format(i) for i in results_data_names])

###### FUNC WRAPPER and CALLBACK #######

def qaoa_wrapper(next_point):
    fin_state, mean_energy, variance, fidelity = qaoa.quantum_algorithm(next_point)
    approx_ratio = mean_energy/qaoa.gs_en
    
    return mean_energy
    
best_energy = 10
best_fidelity = 0
best_approx_ratio = 0

def callbackF(point):
    global best_energy, best_fidelity, best_approx_ratio 
    
    fin_state, mean_energy, variance, fidelity = qaoa.quantum_algorithm(point)
    approx_ratio = mean_energy/qaoa.gs_en
    
        
    if mean_energy < best_energy:
        best_energy = mean_energy
    if fidelity > best_fidelity:
        best_fidelity = fidelity
    if approx_ratio > best_approx_ratio:
        best_approx_ratio = approx_ratio
        
    new_data = ([len(data_)] +
                 point.tolist() +
                 [
                 mean_energy,
                 best_energy,
                 approx_ratio,
                 best_approx_ratio,
                 fidelity,
                 best_fidelity,
                 variance
                 ]
               )
    data_.append(new_data)
    print(f'{len(data_)}/{nbayes}, en: {mean_energy}, var: {variance}, '
          f'fid: {fidelity},\n {point}')
    
    df = pd.DataFrame(data = data_, columns = results_data_names)
    df.to_csv(folder + "/" + file_name , columns = results_data_names, header = data_header)

#####  OPTIMIZATION   #############

bounds = param_range.tolist()*depth
init_point = np.random.uniform(param_range[0, 0], param_range[0,1], 2*depth)
results = minimize(qaoa_wrapper, 
                   x0 = init_point,
                   method = optimizer_method,
                   callback = callbackF,
                   bounds = bounds)

info_file_name = folder + "/" + 'info.csv'
res = {}
for i in ['fun', 'nfev', 'message', 'x']:
    res[i] = results[i]
print(res)
df = pd.DataFrame.from_dict(res)
df.to_csv(info_file_name)


