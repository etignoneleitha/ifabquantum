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
from sklearn.metrics.pairwise import euclidean_distances

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
shots = args.shots
gate_noise = args.gate_noise
kernel_optimizer = 'fmin_l_bfgs_b' #fmin_l_bfgs_b'#'differential_evolution' #fmin_l_bfgs_b'
diff_evol_func = None
method = 'DIFF-EVOL'
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


qaoa = qaoa_qutip(G, 
                  shots = shots, 
                  gate_noise = gate_noise,
                  problem = problem)





gs_energy, gs_state, degeneracy = qaoa.gs_en, qaoa.gs_states, qaoa.deg

print('Information on the hamiltonian')
print('GS energy: ',gs_energy)
print('GS binary:', qaoa.gs_binary)
print('GS degeneracy: ', degeneracy)

print('GS :', qaoa.gs_states[0])
print('ham :, ', qaoa.H_c)

DEFAULT_PARAMS["seed"] = seed + i_trial

########### CREATE GP AND FIT TRAINING DATA  #####################


const_kern = ConstantKernel(
                            constant_value = 1.1,
                            constant_value_bounds = DEFAULT_PARAMS['constant_bounds']
                            )
mat_kern = Matern(
                  length_scale=0.11, 
                  length_scale_bounds=DEFAULT_PARAMS['length_scale_bounds'], 
                  nu=1.5
                  )
kernel = const_kern * mat_kern 

if shots is not None or gate_noise is not None:
    kernel += WhiteKernel(noise_level=0.5)

gp = MyGaussianProcessRegressor(kernel=kernel,
                                optimizer= kernel_optimizer, #fmin_l_bfgs_bor differential_evolution
                                angles_bounds=param_range,
                                n_restarts_optimizer=9,
                                gtol=1e-6,
                                max_iter=1e4,
                                diff_evol_func = diff_evol_func
                                )

print('Created gaussian process istance with starting kernel')
print(gp.kernel)
X_train, y_train = qaoa.generate_random_points(nwarmup, depth, param_range)

print('Random generated X train:', X_train)
print('With energies: ', y_train)
print('\n\n\n')
gp.fit(X_train, y_train)
print(gp.get_covariance_matrix())

#gp.get_acquisition_function(show = True, save = False)
print('Just fitted data_ so now we have kernel and kernel_: ')
print(gp.kernel)
print(gp.kernel_)

print('\n\n')
print('Now covariance LL^\dag:')
print('or')
print(gp.get_covariance_matrix_cholesky())
print('\ncompared to')
print(gp.get_covariance_matrix())



############### CREATE DATA ###############

def angle_names_string():
    gamma_names = [f'GAMMA_{i}' for i in range(depth)]
    beta_names = [f'BETA_{i}' for i in range(depth)]
    
    angle_names = []
    for i in range(depth):
        angle_names.append(beta_names[i])
        angle_names.append(gamma_names[i])
        
    return angle_names
        
output_folder = Path(__file__).parents[1] / "output"
file_name = f'Bayes_{problem}_p_{depth}_num_nodes_{num_nodes}_train_{nbayes}_seed_{seed}'
if shots is not None:
    file_name += f'_shots_{shots}'
if gate_noise is not None:
    file_name += f'_noise_{gate_noise}'

file_name += '.dat'
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
                                 'corr_length',
                                 'const_kernel',
                                 'std_energies',
                                 'average_distances',
                                 'nit',
                                 'time_opt_bayes',
                                 'time_qaoa',
                                 'time_opt_kernel',
                                 'time_step'
                                 ]
if len(gp.kernel_.theta) > 2:
    results_data_names += ['noise_kernel']
                     
for i_tr, x in enumerate(X_train):
    fin_state, mean_energy, variance, fidelity_tot =  qaoa.quantum_algorithm(x)
    approx_ratio = mean_energy/qaoa.gs_en
    
    if i_tr == 0:
        best_energy = mean_energy
        best_fidelity = fidelity_tot
        best_approx_ratio = approx_ratio
    
    else:
        if mean_energy < best_energy:
            best_energy = mean_energy
        if fidelity_tot > best_fidelity:
            best_fidelity = fidelity_tot
        if approx_ratio > best_approx_ratio:
            best_approx_ratio = approx_ratio
            
    new_data = [i_tr] +\
                 x + \
                 [mean_energy,
                  best_energy,
                  approx_ratio,
                  best_approx_ratio,
                  fidelity_tot,
                  best_fidelity,
                  variance,
                  np.exp(gp.kernel_.theta[1]),
                  np.exp(gp.kernel_.theta[0]),
                  0, 0, 0, 0, 0, 0, 0]
    
    if len(gp.kernel_.theta) > 2:
        new_data += [np.exp(gp.kernel_.theta[2])]
    
    data_.append(new_data)
                
print('groundstate :',qaoa.gs_en)
folder_name = file_name.split('.')[0]
folder = os.path.join(output_folder, folder_name)
os.makedirs(folder, exist_ok = True)
data_header = " ".join(["{:>7} ".format(i) for i in results_data_names])


########    BAYESIAN OPTIMIZATION   #############


print('Training ...')
log_likelihood_grids = []
kernel_opts = []
kernel_matrices = []
acq_funcs = []

for i in range(nbayes):
    
    start_time = time.time()
    
    ### ONE STEP BAYES OPT ####
    next_point, n_it, avg_sqr_distances, std_pop_energy = gp.bayesian_opt_step(method)
    bayes_time = time.time() - start_time
    
    ### EVALUATE QAOA AT NEXT POINT####
    fin_state, mean_energy, variance, fidelity_tot = qaoa.quantum_algorithm(next_point)
    qaoa_time = time.time() - start_time - bayes_time
    y_next_point = mean_energy
    fidelity = fidelity_tot
    approx_ratio = mean_energy/qaoa.gs_en
    # if depth <2:
#         acq_func = gp.get_acquisition_function(show = False, save = True)
#         acq_funcs.append(acq_func)
    #log_marginal_likelihood_grid = gp.get_log_marginal_likelihood_grid(show = False, save = True)
    
    #log_likelihood_grids.append(log_marginal_likelihood_grid)
   # k_matrix, _ = gp.get_covariance_matrix()
    #kernel_matrices.append(k_matrix)
    
    #kernel_opts.append(gp.samples)
        
        
    #### FIT NEW POINT #####
    gp.fit(next_point, y_next_point)
    
    params = np.exp(gp.kernel_.theta)
    constant_kernel = params[0]
    corr_length = params[1]
    if len(params) > 2:
        noise_kernel = params[2]
    
    if mean_energy < best_energy:
        best_energy = mean_energy
    if fidelity_tot > best_fidelity:
        best_fidelity = fidelity_tot
    if approx_ratio > best_approx_ratio:
        best_approx_ratio = approx_ratio
        
    kernel_time = time.time() - start_time - qaoa_time - bayes_time
    print('\nKernel', gp.kernel_)
    step_time = time.time() - start_time
    
    new_data = ([i + 1] +
                 next_point +
                 [
                 y_next_point,
                 best_energy,
                 approx_ratio,
                 best_approx_ratio,
                 fidelity,
                 best_fidelity,
                 variance,
                 corr_length,
                 constant_kernel,
                 std_pop_energy,
                 avg_sqr_distances,
                 n_it,
                 bayes_time,
                 qaoa_time,
                 kernel_time,
                 step_time
                 ]
               )
    if len(gp.kernel_.theta) > 2:
        new_data += [noise_kernel]
    data_.append(new_data)
    print(f'{i +1}/{nbayes}, en: {mean_energy}, var: {variance}, '
          f'fid: {fidelity_tot},\n {next_point}')
    
    df = pd.DataFrame(data = data_, columns = results_data_names)
    df.to_csv(folder + "/" + file_name , columns = results_data_names, header = data_header)
    
   #  if diff_evol_func == None:
#         np.save(folder +"/"+ "kernel_opt".format(i), np.array(kernel_opts, dtype = object))
#     else:
#         np.save(folder +"/"+ "opt".format(i), np.array(kernel_opts, dtype = object))
#     np.save(folder +"/"+ "log_marg_likelihoods".format(i), np.array(log_likelihood_grids,  dtype = object))
#     #np.save(folder +"/"+ "kernel_matrices".format(i), np.array(kernel_matrices, dtype = object))
#     np.save(folder +"/"+ "acq_funcs".format(i), np.array(acq_funcs, dtype = object))




############ END BAYES OPT ###################


best_x, best_y, where = gp.get_best_point()
data_.append(data_[where])

df = pd.DataFrame(data = data_, columns = results_data_names)
df.to_csv(folder + "/" + file_name , columns = results_data_names, header = data_header)

print('Best point: ' , data_[where])
print('time: ', time.time() - global_time)


