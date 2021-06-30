import numpy as np
from matplotlib import pyplot as plt
from utils.qaoa import grid_search, QAOA

from utils.default_params import *
from utils.gaussian_proc import bayesian_opt
from utils.graph import create_graph

#Set global parameters
params_bayes = {"N_train": 10,
                "N_test": 100,
                "acq_function": 'EI',
                "num_max_iter_bayes": 100,
                }
penalty = 2
shots = 1000

# Set graph instance & its complement
edge_list = [(0,1), (1,2), (0,2), (2,3), (2,4)]
num_nodes = 5
G, solution_state_str = create_graph(num_nodes, edge_list)
print(solution_state_str)

# points = grid_search(G, 20, penalty=penalty)

results = bayesian_opt(G,
                       penalty,
                       shots,
                       params_bayes=params_bayes)

np.savetxt("results_2.dat", results)

list_states = []

for gamma, beta, _ in results:
    extimated_f1, pretty_counts = QAOA(G,
                                       gamma,
                                       beta,
                                       penalty=DEFAULT_PARAMS["penalty"],
                                       shots=shots,
                                       basis="S")
    if solution_state_str in pretty_counts:
        list_states.append(pretty_counts[solution_state_str] / shots)
    else:
        list_states.append(0.0)

np.savetxt("frequency_solution.dat", list_states)
