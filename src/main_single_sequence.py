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



## to plot the landscape!!

num = 25
a, b, c = qaoa.get_landscape(param_range, num, verbose = 1)
np.savetxt(f'energy_landscape_{param_range[0]}_nodes_{num_nodes}_prob_{problem}_{num}.dat', a)
np.savetxt(f'fidelities_landscape_{param_range[0]}_nodes_{num_nodes}_prob_{problem}_{num}.dat', b)
np.savetxt(f'variances_landscape_{param_range[0]}_nodes_{num_nodes}_prob_{problem}_{num}.dat', c)
exit()

gs_energy, gs_state, degeneracy = qaoa.gs_en, qaoa.gs_states, qaoa.deg

print('Information on the hamiltonian')
print('GS energy: ',gs_energy)
print('GS binary:', qaoa.gs_binary)
print('GS degeneracy: ', degeneracy)

print('GS :', qaoa.gs_states[0])
print('ham :, ', qaoa.H_c)




## to plot a final state!

#for the mis on the chair graph the paramters of best fidelity and lowest energy 
#do not coincide, here they are:
#6 NOdes:
#largest fidelity params:  1.5079644737231006 0.7539822368615503
#lowest energy params: 2.701769682087222 2.6389378290154264

#10 nodes:
#fidelity best: 1.8849555921538756 1.0053096491487339
#lowest energy: 2.701769682087222 2.638937829015426

bit_strings = [str(np.binary_repr(i, num_nodes)) for i in range(2**num_nodes)]

res = qaoa.quantum_algorithm([2.701769682087222, 2.638937829015426])
state_ = res[0].full()
state_probs = np.squeeze(np.abs(state_)**2)
colors = ['green']*(2**num_nodes)
where_1 = int(qaoa.gs_binary[0], 2)
where_2 = int(qaoa.gs_binary[1], 2)
print(where_1, where_2)
colors[where_1] = 'red'
colors[where_2] = 'red'
#colors[]
plt.bar(range(2**num_nodes), state_probs, color = colors)
plt.xticks(ticks =range(0, 2**num_nodes, 10), labels = bit_strings[::10], rotation = 'vertical')
plt.tight_layout()
plt.show()

exit()

