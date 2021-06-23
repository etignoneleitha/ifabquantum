import numpy as np
import networkx as nx
import itertools
import functools
import cmath
from tqdm import tqdm
from numpy.linalg import multi_dot
from scipy.linalg import expm
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../src/')
from scipy.optimize import minimize
from utils import *

'''
This code is used to generate many instances of graphs, apply QAOA with a different beta
at each node of the graph and find the best parameters with scipy.optimize to
see how they distribute
'''


def QAOA_step(H, expH, expB):
    initial_state = np.ones(2**N)*(1/np.sqrt(2**N))          #init = 1/4(1, 1, 1, ..., 1)
    final_state = np.dot(expB, np.dot(expH, initial_state))  #final = expB.expH.init
    final_state_conj = np.conj(final_state)
    expected_energy = multi_dot([final_state_conj, H, final_state]) #exp_energy = final_bra.H.final_ket
    probabilities = final_state_conj*final_state
    
    return probabilities, expected_energy

def simulate_QAOA_diff_betas(params):
    '''
    applies one step of QAOA, generating the Hamiltonian, the X gates and returning 
    the average value of f1

    Parameters
    ----------
    params : real
        gamma and beta, one beta for each node

    Returns
    -------
    expected_energy : float
        the average value of f1

    '''
    gamma = params[0]
    betas = params[1:]
    H = np.diag(energies)
    expH = np.diag([cmath.exp((-gamma)*en*1j) for en in energies])
    Xs = [expm(-1j*betas[i]*np.array([[0,1],[1,0]])) for i in range(N)]
    expB = functools.reduce(np.kron, Xs)

    #Execute QAOA
    prob, expected_energy = QAOA_step(H, expH, expB)
    
    return expected_energy

def simulate_QAOA_fixed_beta(params):
    '''
    applies one step of QAOA, generating the Hamiltonian, the X gates and returning 
    the average value of f1

    Parameters
    ----------
    params : real
        gamma and beta
        
    Returns
    -------
    expected_energy : float
        the average value of f1

    '''
    gamma = params[0]
    beta = params[1]
    H = np.diag(energies)
    expH = np.diag([cmath.exp((-gamma)*en*1j) for en in energies])
    Xs = [expm(-1j*beta*np.array([[0,1],[1,0]])) for i in range(N)]
    expB = functools.reduce(np.kron, Xs)

    #Execute QAOA
    prob, expected_energy = QAOA_step(H, expH, expB)
    
    return expected_energy

running_params = []
def callbackF(X):
   running_params.append(X)
omega = 2

N = 8
p = 0.4
number_of_graphs = 1
callback = True
if number_of_graphs> 3:
    callback = False
graphs = []
seeds = []

optimal_params = []
data = []
for i in tqdm(range(number_of_graphs)):
    seed = np.random.randint(10000)
    G = create_random_graph(N, p, seed)
    G_comp = nx.complement(G)
    binary_combs = np.array(list(itertools.product([0, 1], repeat=N)))
    
    hamiltonian = 'maxcut'
    if hamiltonian == 'maxclique':
        energies = [max_clique_cost(G_comp, omega, comb) for comb in binary_combs]
        real_max_clique = classical_solution(G, hamiltonian, omega = 2, verbose = 1)
    if hamiltonian == 'maxcut':
        energies = [max_cut_cost(G, comb) for comb in binary_combs] 
    
    #plt.hist(energies)
    
    #We use scopy.optimize with the COBYLA method (seems to be most efficient for constrained
    #problems), using boundaries [0, np.pi] for every variable.
    if callback:
        best_params = minimize(simulate_QAOA_diff_betas, 
                               x0 =np.random.uniform(size = N+1),
                               callback = callbackF,
                               options = {'maxiter' : 10000}, 
                               )
    else:
        best_params = minimize(simulate_QAOA_diff_betas, 
                               x0 =np.random.uniform(size = N+1),
                               options = {'maxiter' : 10000}
                               )
    #best_params_fixed = minimize(simulate_QAOA_fixed_beta, x0 = np.random.uniform(size = 2), method = 'COBYLA')
    seeds.append(seed)
    
    if best_params.success:
        best_betas = np.array(best_params.x, dtype = 'float')
        best_energy = best_params.fun
        if hamiltonian == 'maxclique':
            data.append([seed, best_betas, best_energy, real_max_clique.values()])
    else:
        print('At iteration {} process failed'.format(i))
    
#np.savetxt('../data/processed/{}/Different_betas_optimal_params_N8_p04.dat'.format(hamiltonian),data, fmt = '%s')


if callback:
    fig = plt.figure()
    labels = ['gamma'] + ['beta {}'.format(i) for i in range(N)]
    lines = plt.plot(running_params)
    [lines[i].set_label(labels[i]) for i in range(N)]
    plt.legend()
    plt.title('Evolution of params, final en {}'.format(best_energy))
    plt.xlabel('Time steps')
    plt.ylabel('Angles')




