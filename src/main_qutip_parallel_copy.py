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
                     'corr_length ',
                     'const_kernel ',
                     'std_energies ',
                     'average_distances ',
                     'nit ',
                     'time_opt_bayes ',
                     'time_qaoa ',
                     'time_opt_kernel ',
                     'time_step '
                     ]

### CREATE GRAPH
# pos = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [0, 4]])
# pos = np.array([[0, 1], [1, 2], [3, 2], [0, 3], [0, 4], [0,5]])
np.random.seed(seed)
num_graph = seed

# name_plot = str(seed)
# G = create_random_regular_graph(8, seed=seed, name_plot=name_plot) #create_chair_graph()
# name_plot = "chair"
# G = create_chair_graph(name_plot="chair")

name_plot = str(seed)
G = create_random_regular_graph(num_nodes, degree=3, seed=1)

qaoa = qaoa_qutip(G, problem="MAX-CUT")
gs_energy, gs_state, degeneracy = qaoa.gs_en, qaoa.gs_states, qaoa.deg

print('Information on the hamiltonian')
print('GS energy: ',gs_energy)
print('GS degeneracy: ', degeneracy)
print('GS: ', qaoa.gs_binary, '\n\n\n')

DEFAULT_PARAMS["seed"] = seed + i_trial
output_folder = Path(__file__).parents[1] / "output"
file_name = f'lfgbs_p_{depth}_punti_{nwarmup + nbayes}_warmup_{nwarmup}_train_{nbayes}_trial_{i_trial}_graph_{name_plot}.dat'
data = []
### CREATE GP AND FIT TRAINING DATA
# kernel = ConstantKernel(1)*RBF(0.2, length_scale_bounds = (1E-1, 1E2))
kernel = ConstantKernel(1.1, constant_value_bounds = DEFAULT_PARAMS['constant_bounds']) *\
             Matern(length_scale=0.11, length_scale_bounds=DEFAULT_PARAMS['length_scale_bounds'], nu=1.5)
gp = MyGaussianProcessRegressor(kernel=kernel,
                                optimizer= DEFAULT_PARAMS['kernel_optimizer'], #fmin_l_bfgs_bor differential_evolution
                                #optimizer='differential_evolution', #fmin_l_bfgs_bor
                                angles_bounds=param_range,
                                n_restarts_optimizer=0,
                                gtol=1e-6,
                                max_iter=1e4
                                )

print('Created gaussian process istance with starting kernel')
print(gp.kernel)
X_train, y_train = qaoa.generate_random_points(nwarmup, depth, param_range)


print('Random generated X train:', X_train)
print('With energies: ', y_train)
print('\n\n\n')

gp.fit(X_train, y_train)

print('Just fitted data so now we have kernel and kernel_: ')
print(gp.kernel)
print(gp.kernel_)

print('\n\n')
print('So starting cholesky is')
print(gp.L_)
print('and covariance LL^\dag:')
print(gp.L_ @ np.transpose(gp.L_))
print('\n\n')
print('REPRODUCING ERROR\n')
gp.kernel_.theta = [2.2945, -0.8897]
print('with kernel')
print(gp.kernel_)
print('\n and matrix')
print(gp.get_covariance_matrix())
gp.predict([[0.8562, 0.256 ]], return_std = True)


exit()






