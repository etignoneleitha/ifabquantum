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
 
from utils import *


'''
We calculate the expected energy of QAOA after applying one layer, p = 1, with two methods:
-- random gamma and one random beta for each node
-- random gamma and one beta (classic approach to QAOA)
We repeat it N times to see how the energy distributes according to the
betas and then compare it to the standard result with only one beta
'''

E = []
E = [(0,1), (1,2), (2,3), (0,2)]
G, N= create_graph_from_edges(E)
G_comp = nx.complement(G)
omega = 2
p = 1

binary_combs = np.array(list(itertools.product([0, 1], repeat=N)))  
hamiltonian = 'maxclique'
if hamiltonian == 'maxclique':
    energies = [max_clique_cost(G_comp, omega, comb) for comb in binary_combs]
if hamiltonian == 'maxcut':
    energies = [max_cut_cost(G_comp, comb) for comb in binary_combs] 

#First run with different betas on each node

all_gammas = []
all_betas = []
exp_energies = []
all_probabilities = []

for i in tqdm(range(1000)):  #for 10000 iterations about 25 seconds

    gamma = np.random.uniform(0, 2*np.pi)
    betas = np.random.uniform(0, np.pi, N)
    
    H = np.diag(energies)
    expH = np.diag([cmath.exp((-gamma)*en*1j) for en in energies])
    Xs = [expm(complex(0, -betas[i])*np.array([[0,1],[1,0]])) for i in range(N)]
    expB = functools.reduce(np.kron, Xs)

    #Execute QAOA
    initial_state = np.ones(2**N)*(1/np.sqrt(2**N))          #init = 1/4(1, 1, 1, ..., 1)
    final_state = np.dot(expB, np.dot(expH, initial_state))  #final = expB.expH.init
    final_state_conj = np.conj(final_state)
    expected_energy = multi_dot([final_state_conj, H, final_state]) #exp_energy = final_bra.H.final_ket
    probabilities = final_state_conj*final_state
    
    all_gammas.append(gamma)
    all_betas.append(betas)
    exp_energies.append(expected_energy)
    all_probabilities.append(probabilities)

print('Energy of every possible configuration: \n')
print(energies)
#fig = plt.figure()
#plt.hist(energies)
#plt.title('Real energies')


fig = plt.figure()
n, bins, pat = plt.hist(np.real(exp_energies))
plt.title('Energies (with {} random parameters)'.format(len(betas) + 1))
print('Energy: Bins {} \n N_bins {}'.format(bins, n))

fig = plt.figure()
n, bins, pat = plt.hist(np.array(np.real(all_probabilities))[:,2])
plt.title('Probability correct state 1110 (diff betas): ')
print('Proabbility: Bins {} \n N_bins {}'.format(bins, n))

all_gammas_2 = []
all_betas_2 = []
exp_energies_2 = []
all_probabilities_2 = []

#Second run with only one beta
for i in tqdm(range(1000)):  #for 10000 iterations about 25 seconds

    gamma = np.random.uniform(0, 2*np.pi)
    beta = np.random.uniform(0, np.pi)
    
    H = np.diag(energies)
    expH = np.diag([cmath.exp((-gamma)*en*1j) for en in energies])

    Xs = [expm(complex(0, -beta)*np.array([[0,1],[1,0]])) for i in range(N)]
    expB = functools.reduce(np.kron, Xs)

    #Execute QAOA
    initial_state = np.ones(2**N)*(1/np.sqrt(2**N))
    final_state = np.dot(expB, np.dot(expH, initial_state))
    final_state_conj = np.conj(final_state)
    expected_energy = multi_dot([final_state_conj, H, final_state])
    probabilities = final_state_conj*final_state
    
    all_gammas_2.append(gamma)
    all_betas_2.append(beta)
    exp_energies_2.append(expected_energy)
    all_probabilities_2.append(probabilities)

fig = plt.figure()
n, bins, pat = plt.hist(np.real(exp_energies_2))
plt.title('Energies with fixed beta')
print('Energy: Bins {} \n N_bins {}'.format(bins, n))

fig = plt.figure()
plt.title('Probability of the correct state 1110 (beta fixed): ')
n, bins, pat = plt.hist(np.array(all_probabilities_2)[:,2])
print('Probability: Bins {} \n N_bins {}'.format(bins, n))
    
