import numpy as np
from qutip import *
from pulser.devices import Chadoq2
import networkx as nx
import matplotlib.pyplot as plt


'''
Calculates the GS of H = \Omega \sum_i \sigma^x_i - \delta \sum_i \sigma^z_i + \sum_{ij} n_i n_j
'''

def list_operator(N_qubit, op):
    ''''
    returns a a list of tensor products with op on site 0, 1,2 ...
    '''
    op_list = []

    for n in range(N_qubit):
        op_list_i = []
        for m in range(N_qubit):
            op_list_i.append(qeye(2))

        op_list_i[n] = op
        op_list.append(tensor(op_list_i))

    return op_list


def calculate_physical_gs(G, p = 2):
    '''
    returns grounstate and energy
    '''
    N_qubit=len(G.nodes())

    sx_list = list_operator(N_qubit, sigmax())
    sz_list = list_operator(N_qubit, sigmaz())

    H=0
    for n in range(N_qubit):
        H +=  sz_list[n]
    for i, edge in enumerate(G.edges):
        H += 2*p * sz_list[edge[0]]*sz_list[edge[1]]
        H -= p * sz_list[edge[0]]
        H -= p * sz_list[edge[1]]
    energies, eigenstates = H.eigenstates(sort = 'low')
    print(H.diag())
    if energies[0] ==  energies[1]:
        print('DEGENERATE GROUND STATE')
        deg = True
        gs_en = energies[:2]
        gs_state = eigenstates[:2]
    else:
        deg = False
        gs_en = energies[0]
        gs_state = eigenstates[0]
    return gs_en, gs_state, deg


def cost_op_calculation(G, penalty = 10):

    N_qubit = len(G.nodes())
    sz_list = list_operator(N_qubit, sigmaz())

    H_cost = Qobj()
    for n in range(N_qubit):
        H_cost -= sz_list[n]
    for edge in G.edges:
        H_cost +=  penalty*sz_list[edge[0]]*sz_list[edge[1]]

    return H_cost

def list_operator(N_qubit, op):
    ''''
    returns a a list of tensor products with op on site 0, 1,2 ...
    '''
    op_list = []

    for n in range(N_qubit):
        op_list_i = []
        for m in range(N_qubit):
            op_list_i.append(qeye(2))

        op_list_i[n] = op
        op_list.append(tensor(op_list_i))

    return op_list

def calculate_physical_gs(G, p = 1, omega = 1, delta = 1, U = 10):
    '''
    returns groundstate and energy
    '''
    N_qubit=len(list(G))
    Omega=omega/2
    delta=delta/2

    ## Defining lists of tensor operators
    ni = (qeye(2) - sigmaz())/2

    sx_list = list_operator(N_qubit, sigmax())
    sz_list = list_operator(N_qubit, sigmaz())
    ni_list = list_operator(N_qubit, ni)

    H=0
    for n in range(N_qubit):
        #H += Omega*sx_list[n]
        H -= delta * ni_list[n]
    for i, edge in enumerate(G.edges):
        H +=  U[i]*ni_list[edge[0]]*ni_list[edge[1]]
    energies, eigenstates = H.eigenstates(sort = 'low')
    if energies[0] ==  energies[1]:
        print('DEGENERATE GROUND STATE')
        deg = True
        gs_en = energies[:2]
        gs_state = eigenstates[:2]
    else:
        deg = False
        gs_en = energies[0]
        gs_state = eigenstates[0]

    return gs_en, gs_state, deg

def main():
    U = 10
    G = nx.Graph()
    G.add_edges_from([[0,1], [0,2], [0,3], [0,4],[1, 2]])
    print(G.edges())
    gs_en, gs_state, deg = calculate_physical_gs(G, p = 2)
    print('Ground state energy and state: ',gs_en, gs_state)

    Cost_operator = cost_op_calculation(G, penalty = 10)
    exp = expect(Cost_operator, gs_state)
    #print(np.array(gs_state)**2)

    print('Expected classical cost:', exp)

main()


