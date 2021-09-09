import numpy as np
import sys
from utils.qaoa import grid_search, QAOA, evaluate_cost, str2list

from utils.default_params import *
from utils.gaussian_proc import bayesian_opt
from utils.graph import (create_graph, 
                         create_random_graphs,
                         load_graph_pickle
                         )

#Set global parameters
params_bayes = {"N_train": 10,
                "N_test": 100,
                "acq_function": 'EI',
                "num_max_iter_bayes": 100,
                }
penalty = 2
shots = 1000
if len(sys.argv) > 1:
    key_graph = int(sys.argv[1])
else:
   key_graph = np.random.randint(100000)
print(key_graph)

G, solution_state_str = load_graph_pickle(key_graph)

# G, solution_state_str = create_graph(len(graph.nodes), graph.edges)
min_energy = evaluate_cost(G, str2list(solution_state_str), basis = 'S')
#print('Solution: {} with energy: {}'.format(solution_state_str, min_energy))

# points = grid_search(G, 20, penalty=penalty)

# run average: 3*n_iter array with average energy, best state prob,
# approx ratio for every iteration
# all_run_average: a list of run_average for different n. of initial points

N_average = 40
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
        filename = ("../data/raw/graph_{}_"
                    "N_train_{}_"
                    "iter_{}_"
                    "N_average_{}.dat".format(key_graph, n_start, run_i, N_average)
                    )

        np.savetxt(filename, run_data)
        #np.savetxt("../data/raw/graph_{key_graph}_N_train_{}_iter_{}_N_average_{}.dat".format(key_graph, n_start, run_i), run_data)


