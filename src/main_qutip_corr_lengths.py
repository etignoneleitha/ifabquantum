import sys
sys.settrace
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
from utils.qaoa_qutip import *
from utils.gaussian_process import *
from utils.create_graphs import (create_random_graph,
                                 create_chair_graph,
                                 create_random_regular_graph
                                 )
from utils.parameters import parse_command_line
from utils.default_params import *
import time
from pathlib import Path
import os
from sklearn.metrics.pairwise import euclidean_distances

np.set_printoptions(precision=4, suppress=True)


### READ PARAMETERS FROM COMMAND LINE
args = parse_command_line()

seed = args.seed

fraction_warmup = args.fraction_warmup
depth = args.p
trials = args.trials
i_trial = args.i_trial
num_nodes = args.num_nodes
nwarmup = args.nwarmup
average_connectivity = args.average_connectivity
diff_evol_func = args.diff_evol_func


if diff_evol_func == None:
    kernel_optimizer = 'fmin_l_bfgs_b'
else:
    kernel_optimizer = None

### PARAMETERS
nbayes = args.nbayes
method = 'DIFF-EVOL'
param_range = np.array([[0.01, np.pi], [0.01, np.pi]])   # extremes where to search for the values of gamma and beta

global_time = time.time()
white_spaces = " " * (6 * 2 * depth)
results_structure = ['iter ',
                     white_spaces + 'point' + white_spaces,
                     'energy   ',
                     'fidelity  ',
                     'variance  ',
                     'corr_length  ',
                     'const_kernel  ',
                     'std_energies  ',
                     'average_distances  ',
                     'nit  ',
                     'time_opt_bayes  ',
                     'time_qaoa  ',
                     'time_opt_kernel  ',
                     'time_step  '
                     ]

### CREATE GRAPH
# pos = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [0, 4]])
# pos = np.array([[0, 1], [1, 2], [3, 2], [0, 3], [0, 4], [0,5]])
np.random.seed(seed)
num_graph = seed

# name_plot = str(seed)
# G = create_random_regular_graph(8, seed=seed, name_plot=name_plot) #create_chair_graph()


name_plot = str(seed)
#G = create_random_regular_graph(num_nodes, degree=3, seed=1)
G = create_chair_graph(name_plot="chair")

qaoa = qaoa_qutip(G, problem="MIS")
gs_energy, gs_state, degeneracy = qaoa.gs_en, qaoa.gs_states, qaoa.deg

print('Information on the hamiltonian')
print('GS energy: ',gs_energy)
print('GS degeneracy: ', degeneracy)
print('GS: ', qaoa.gs_binary, '\n\n\n')

DEFAULT_PARAMS["seed"] = seed + i_trial
output_folder = Path(__file__).parents[1] / "output_cluster"
file_name = f'lfgbs_p_{depth}_punti_{nwarmup + nbayes}_warmup_{nwarmup}_train_{nbayes}_trial_{i_trial}_graph_{name_plot}.dat'
data = []
### CREATE GP AND FIT TRAINING DATA
# kernel = ConstantKernel(1)*RBF(0.2, length_scale_bounds = (1E-1, 1E2))
kernel = ConstantKernel(1.1, constant_value_bounds = DEFAULT_PARAMS['constant_bounds']) *\
             Matern(length_scale=[0.11]*depth*2, length_scale_bounds=DEFAULT_PARAMS['length_scale_bounds'], nu=1.5)
gp = MyGaussianProcessRegressor(kernel=kernel,
                                optimizer= kernel_optimizer, #fmin_l_bfgs_bor differential_evolution
                                #optimizer='differential_evolution', #fmin_l_bfgs_bor
                                angles_bounds=param_range,
                                n_restarts_optimizer=0,
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

exit()

fig = plt.figure()
distances = euclidean_distances(X_train)
#ind = np.diag_indices_from(distances)
#distances[ind] = y_train
print(distances)
print(X_train[2],'\n', X_train[6],'\n', X_train[7],'\n', X_train[8])
gp.fit(X_train, y_train)
print(gp.get_covariance_matrix())
#log_marginal_likelihood_grid = gp.get_log_marginal_likelihood_grid()


print('Just fitted data so now we have kernel and kernel_: ')
print(gp.kernel)
print(gp.kernel_)

print('\n\n')
print('Now covariance LL^\dag:')
print('or')
print(gp.get_covariance_matrix_cholesky())
print('\ncompared to')
print(gp.get_covariance_matrix())
print('REPRODUCING ERROR\n')


data = []
for i_tr, x in enumerate(X_train):
    fin_state, mean_energy, variance, fidelity_tot =  qaoa.quantum_algorithm(x)
    data.append([i_tr]
                + x
                + [y_train[i_tr],
                fidelity_tot,
                variance]+
                np.exp(gp.kernel_.theta[1:]).tolist() +
                [np.exp(gp.kernel_.theta[0]),
                0, 0, 0, 0, 0, 0, 0]
                )
print(len(data[0]))
### BAYESIAN OPTIMIZATION
print('Training ...')
for i in range(nbayes):
    
    start_time = time.time()
    next_point, n_it, avg_sqr_distances, std_pop_energy = gp.bayesian_opt_step(method)
    
    fin_state, mean_energy, variance, fidelity_tot = qaoa.quantum_algorithm(next_point)
    bayes_time = time.time() - start_time
    y_next_point = mean_energy
    qaoa_time = time.time() - start_time - bayes_time
    fidelity = fidelity_tot
    gp.fit(next_point, y_next_point)
    #constant_kernel, corr_length = np.exp(gp.average_kernel_params)
    if diff_evol_func == None:
        params = np.exp(gp.kernel_.theta)
    else:
        params = np.exp(gp.average_kernel_params)
    constant_kernel = params[0]
    corr_length = params[1:]
    print(params)
    kernel_time = time.time() - start_time - qaoa_time - bayes_time
    print('now kernel is:')
    print(gp.kernel_)
    step_time = time.time() - start_time
    
    new_data = ([i + nwarmup]
                + next_point
                + [y_next_point,
                fidelity,
                variance] +
                corr_length.tolist()+
                [constant_kernel,
                std_pop_energy,
                avg_sqr_distances,
                n_it,
                bayes_time,
                qaoa_time,
                kernel_time,
                step_time
                ])

    data.append(new_data)
    print(i + nwarmup,'/', nbayes, mean_energy, variance, fidelity_tot, *next_point)

    format_list = ['%+.8f '] * len(new_data)
    format_list[0] = '% 4d '
    format_list[-5] = '% 8d '
    fmt_string = "".join(format_list)
    
    folder_name = file_name.split('.')[0]
    folder = os.path.join(output_folder, folder_name)
    os.makedirs(folder, exist_ok = True)
    
    np.savetxt(folder +"/"+ file_name, data, fmt = fmt_string, header  ="".join(results_structure))
    #np.savetxt(folder +"/"+ "step_{}_kernel_opt.dat".format(i), gp.samples)
    np.savetxt(folder +"/"+ "step_{}_opt.dat".format(i), gp.mcmc_samples)

best_x, best_y, where = gp.get_best_point()
print('ther where:', where, best_x, best_y)
exit()
data.append(data[where])

np.savetxt(folder +"/"+ file_name,
           np.array(data),
           fmt=fmt_string,
           header="".join(results_structure)
           )
print('Best point: ' , data[where])
print('time: ', time.time() - global_time)