import numpy as np
from utils.qaoa import grid_search, QAOA, evaluate_cost, str2list

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
min_energy = evaluate_cost(G, str2list(solution_state_str), basis = 'S')
print('Solution: {} with energy: {}'.format(solution_state_str, min_energy))

# points = grid_search(G, 20, penalty=penalty)

# run average: 3*n_iter array with average energy, best state prob,
# approx ratio for every iteration
# all_run_average: a list of run_average for different n. of initial points

N_average = 20
all_run_average = []
for n_start in [1, 2, 5, 10]:  
    params_bayes['N_train'] = n_start
    
    for run_i in range(N_average):   
        print(n_start, run_i)
        results = bayesian_opt(G,
                               penalty,
                               shots,
                               params_bayes=params_bayes,
                               verbose = False)
        
        run_data = []
        for gamma, beta, _ in results:
            extimated_f1, pretty_counts = QAOA(G,
                                               gamma,
                                               beta,
                                               penalty=DEFAULT_PARAMS["penalty"],
                                               shots=shots,
                                               basis="S")
            if solution_state_str in pretty_counts:
                prob = pretty_counts[solution_state_str]/shots
            else:
                prob = 0.0
            run_data.append([extimated_f1, extimated_f1/min_energy, prob])
        np.savetxt("../data/raw/N_train_{}_iter_{}.dat".format(n_start, run_i), run_data)
