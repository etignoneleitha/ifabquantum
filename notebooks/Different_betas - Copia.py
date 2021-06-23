import numpy as np
import networkx as nx
import itertools
import functools
import cmath
from numpy.linalg import multi_dot
from scipy.linalg import expm
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../src/')
from scipy.optimize import minimize
 
from utils import *


N = 7
p = 0.4
omega = 2

G = create_random_graph(N, p)
G_comp = nx.complement(G)

binary_combs = np.array(list(itertools.product([0, 1], repeat=N)))
hamiltonian = 'maxclique'
if hamiltonian == 'maxclique':
    energies = [max_clique_cost(G_comp, omega, comb) for comb in binary_combs]
if hamiltonian == 'maxcut':
    energies = [max_cut_cost(G_comp, comb) for comb in binary_combs] 


def simulate_QAOA_diff_betas(params):
    gamma = params[0]
    betas = params[1:]
    H = np.diag(energies)
    expH = np.diag([cmath.exp((-gamma)*en*1j) for en in energies])
    Xs = [expm(-1j*betas[i]*np.array([[0,1],[1,0]])) for i in range(N)]
    expB = functools.reduce(np.kron, Xs)

    #Execute QAOA
    initial_state = np.ones(2**N)*(1/np.sqrt(2**N))          #init = 1/4(1, 1, 1, ..., 1)
    final_state = np.dot(expB, np.dot(expH, initial_state))  #final = expB.expH.init
    final_state_conj = np.conj(final_state)
    expected_energy = multi_dot([final_state_conj, H, final_state]) #exp_energy = final_bra.H.final_ket
    probabilities = final_state_conj*final_state
    
    return expected_energy

gamma = np.random.uniform()
betas = np.random.uniform(size = (N,))
best_params = minimize(simulate_QAOA_diff_betas, np.random.uniform(size = N+1), 
                       bounds = np.tile([0,1], (N+1,1))*np.pi)
print(best_params)
print(simulate_QAOA_diff_betas(best_params.x))





def generate_beta_statistics(depth, N_graphs, omega, N_nodes, prob_link, hamiltonian = 'maxclique'):
    '''

    Parameters
    ----------
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    binary_combs = np.array(list(itertools.product([0, 1], repeat=N_nodes)))
    en = []
    seeds = []
    for i in tqdm(range(N_graphs)):
        seed = np.random.randint(1000)
        seeds.append(seed)
        G = create_random_graph(N_nodes, prob_link, seed)
        G_comp = nx.complement(G)
        
        beta = np.random.uniform(size = (N_nodes,))
        gamma = np.random.uniform()
        ens = simulate_QAOA_diff_betas( gamma, beta)
        en.append(ens)
    return en