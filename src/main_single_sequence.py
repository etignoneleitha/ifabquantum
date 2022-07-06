import sys
sys.settrace
import numpy as np
import matplotlib.pyplot as plt
from utils.qaoa_qutip_mod import *
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


a, b, first_exc = qaoa.classical_solution(save = True)

## to plot the landscape!!

# num = 25
# a, b, c = qaoa.get_landscape(param_range, num, verbose = 1)
# np.savetxt(f'energy_landscape_{param_range[0]}_nodes_{num_nodes}_prob_{problem}_{num}.dat', a)
# np.savetxt(f'fidelities_landscape_{param_range[0]}_nodes_{num_nodes}_prob_{problem}_{num}.dat', b)
# np.savetxt(f'variances_landscape_{param_range[0]}_nodes_{num_nodes}_prob_{problem}_{num}.dat', c)
# exit()

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

def print_histogram(res):
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
    

def print_gap_circuit(res):
    states = res[-1]
    
    
    gs_probs = []
    first_exc_probs = []
    for state_ in states:
        print(state_)
        state_ = state_.full()
        state_probs = np.squeeze(np.abs(state_)**2)
        
        prob_gs = 0
        prob_first_excited = 0
        for gs_bitstring in qaoa.gs_binary:
            position_ = int(gs_bitstring, 2)
            prob_gs += state_probs[position_]
        gs_probs.append(prob_gs)
        
        for bitstring in first_exc:
            position_ = int(bitstring, 2)
            prob_first_excited += state_probs[position_]
        first_exc_probs.append(prob_first_excited)
        
    width = 0.2  
    angle_names = ['START', 'Hadamard']
    
    gamma_names = [f'GAMMA_{i}' for i in range(depth)]
    beta_names = [f'BETA_{i}' for i in range(depth)]
    
    
    for i in range(depth):
        angle_names.append(gamma_names[i])
        angle_names.append(beta_names[i])
        
    plt.bar(np.arange(len(states)) - width/2, gs_probs, width, label='GS')
    plt.bar(np.arange(len(states)) + width/2, first_exc_probs, width, label='First excited')
    plt.xticks(ticks = np.arange(0, len(angle_names)), labels = angle_names, rotation= 70)
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.show()

# def print_gap_circuit(res):
#     states = res[-1]
#     
#     print(states)
#     exit()
#     gs_probs = np.abs(states[:,0])**2
#     
#     first_exc_probs = np.abs(states[:,1])**2
#     
#     
#     width = 0.2  
#     angle_names = ['START', 'Hadamard']
#     
#     gamma_names = [f'GAMMA_{i}' for i in range(depth)]
#     beta_names = [f'BETA_{i}' for i in range(depth)]
#     
#     
#     for i in range(depth):
#         angle_names.append(gamma_names[i])
#         angle_names.append(beta_names[i])
#         
#     plt.bar(np.arange(len(states)) - width/2, gs_probs, width, label='GS')
#     plt.bar(np.arange(len(states)) + width/2, first_exc_probs, width, label='First excited')
#     plt.xticks(ticks = np.arange(0, len(angle_names)), labels = angle_names, rotation= 70)
#     plt.ylabel('Probability')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
bit_strings = [str(np.binary_repr(i, num_nodes)) for i in range(2**num_nodes)]
#p=7 [1.2077654287990327,1.8284011657458048,0.35016566578940417,0.951201616422394,0.7615305045116183,3.141592653589793,1.246210168588859,2.419650578888041,2.207402388061187,2.2207929513058935,2.532083050270376,0.9765041597156112,0.7014413717791571,3.0023914514224743]
#p=12 [0.8319522350902833,2.4146713966631164,2.035181005017576,3.118411071634018,1.141240211252959,1.5421021996477782,1.3975433994935895,0.9972914851374582,1.3910028358807496,2.5034532020798426,1.1892923827251118,2.8993615977984484,1.9295227785233275,3.122822851594202,1.8098846256341354,1.2605931098514,2.162126124233499,2.2607234667898846,2.3645444190605427,0.3285713137961852,1.7832318632972628,0.3731896886215793,0.5179208484146487,0.9382976327316992]
#10 nodes graph p=9: [1.7053777443105744,1.4765668080827519,2.255359292874548,1.6298755161424674,0.08924739622605848,1.3840901746142333,2.4409929133292225,2.739292241856487,2.25561990874535,1.476834750885114,1.4496504618667583,2.2534035918318276,1.2183339990649793,1.607834805563501,1.7341204234526684,0.401740985376005,1.5070872181162458,1.398102214694871]
res = qaoa.quantum_algorithm([1.2077654287990327,1.8284011657458048, 0.35016566578940417,0.951201616422394,0.7615305045116183,3.141592653589793,1.246210168588859,2.419650578888041,2.207402388061187,2.2207929513058935,2.532083050270376,0.9765041597156112,0.7014413717791571,3.0023914514224743])

print_gap_circuit(res)


exit()

