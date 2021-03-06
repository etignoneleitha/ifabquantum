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
from scipy.optimize import minimize, basinhopping, differential_evolution, shgo, dual_annealing
np.set_printoptions(precision=4, suppress=True)

'''

Runs through three different global optimizers

'''
    
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
shots = args.shots

param_range = np.array([[0.01, np.pi], [0.01, np.pi]])   # extremes where to search for the values of gamma and beta

global_time = time.time()

np.random.seed(seed)
num_graph = seed
name_plot = str(seed)


################ CREATE GRAPH AND QAOA ################

for depth in [1,2,3,4,5]:
    for optimizer_method in ['basinhopping','diff_evol','dual_annealing']:
        if problem == 'H2' or problem == 'H2_BK':
            G = create_chain(4)
        elif problem == 'H2_reduced' or problem == 'H2_BK_reduced':
            G = create_chain(2)
        else:
            G = create_random_regular_graph(num_nodes, degree=3, seed=1)
    
        qaoa = qaoa_qutip(G, problem=problem, shots = shots)
        gs_energy, gs_state, degeneracy = qaoa.gs_en, qaoa.gs_states, qaoa.deg


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
        file_name = f'{optimizer_method}_{problem}_p_{depth}_num_nodes_{num_nodes}_seed_{seed}'
        if shots is not None:
            file_name += f'_shots_{shots}'
            
        data_iter = []
        data_nfev = []
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
        try:
            os.makedirs(folder, exist_ok = False)
        except:
            continue
        data_header = " ".join(["{:>7} ".format(i) for i in results_data_names])

        ###### FUNC WRAPPER and CALLBACK #######

        best_energy = 10
        best_fidelity = 0
        best_approx_ratio = 0

        def qaoa_wrapper(next_point):
    
            '''This stores every call to the function in the data_nfev list'''
    
            fin_state, mean_energy, variance, fidelity = qaoa.quantum_algorithm(next_point)
            approx_ratio = mean_energy/qaoa.gs_en
    
            global best_energy, best_fidelity, best_approx_ratio 
    
        
            if mean_energy < best_energy:
                best_energy = mean_energy
            if fidelity > best_fidelity:
                best_fidelity = fidelity
            if approx_ratio > best_approx_ratio:
                best_approx_ratio = approx_ratio
        
            new_data = ([len(data_nfev)] +
                         next_point.tolist() +
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
            data_nfev.append(new_data)
    
            df = pd.DataFrame(data = data_nfev, columns = results_data_names)
            df.to_csv(folder + "/" + file_name + '_nfev.dat', columns = results_data_names, header = data_header)
    
            return mean_energy
    


        def callbackF(point):

            '''This stores only one call to the function for each iteration of the
            algorithm and this results are store in the data_iter list'''
    
            fin_state, mean_energy, variance, fidelity = qaoa.quantum_algorithm(point)
            approx_ratio = mean_energy/qaoa.gs_en

            global best_energy, best_fidelity, best_approx_ratio 
    
            if mean_energy < best_energy:
                best_energy = mean_energy
            if fidelity > best_fidelity:
                best_fidelity = fidelity
            if approx_ratio > best_approx_ratio:
                best_approx_ratio = approx_ratio
        
            new_data = ([len(data_iter)] +
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
            data_iter.append(new_data)
            print(f'{len(data_iter)}/{nbayes}, en: {mean_energy}, var: {variance}, '
                  f'fid: {fidelity},\n {point}')
    
            df = pd.DataFrame(data = data_iter, columns = results_data_names)
            df.to_csv(folder + "/" + file_name + '_iterations.dat', columns = results_data_names, header = data_header)
    
            return mean_energy
    

        #####  OPTIMIZATION   #############

        bounds = np.array(param_range.tolist()*depth)
        init_point = np.random.uniform(param_range[0, 0], param_range[0,1], 2*depth)

        # class MyBounds:
        #     def __init__(self, xmax=bounds[:,1], xmin=bounds[:,0]):
        #         self.xmax = np.array(xmax)
        #         self.xmin = np.array(xmin)
        #     def __call__(self, **kwargs):
        #         print(**kwargs)
        #         exit()
        #         x = kwargs["x_new"]
        #         tmax = bool(np.all(x <= self.xmax))
        #         tmin = bool(np.all(x >= self.xmin))
        #         return tmax and tmin
        # 
        # mybounds = MyBounds()

    
        if optimizer_method == 'basinhopping':
    
            def callback_bh(x, fun, conv):
                callbackF(x)
        
            results = basinhopping(qaoa_wrapper,
                                   x0 = init_point,
                                   callback = callback_bh,
                                   niter_success = 2,
                                   )
                                   
            print(results.message)
        elif optimizer_method == 'diff_evol':
            def callback_de(x, convergence):
                callbackF(x)
        
            results = differential_evolution(qaoa_wrapper,
                                            bounds = bounds,
                                            callback = callback_de
                                            )
            print(results.message)
        elif optimizer_method == 'shgo':
            op = {'maxiter':200*depth}
            results = shgo(qaoa_wrapper,
                            bounds = bounds,
                            callback = callbackF,
                            options = op
                                   )
        elif optimizer_method == 'dual_annealing':
            def callback_da(x, e, context):
                callbackF(x)
        
            results = dual_annealing(qaoa_wrapper,
                                    bounds = bounds,
                                    callback = callback_da,
                                    maxfun = 200*depth,
                           
                                   )
        else:
            results = minimize(qaoa_wrapper, 
                               x0 = init_point,
                               method = optimizer_method,
                               callback = callbackF,
                               bounds = bounds)




