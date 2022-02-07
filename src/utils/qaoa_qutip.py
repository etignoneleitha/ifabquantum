import networkx as nx
from collections import Counter, defaultdict, namedtuple
from itertools import product
# import pandas as pd
import random
import numpy as np

# QUANTUM
#from qiskit import Aer, QuantumCircuit, execute
import qutip as qu
from  utils.default_params import *

# VIZ
from matplotlib import pyplot as plt
from utils.default_params import *

from scipy.stats import qmc


class qaoa_qutip(object):

    def __init__(self, G, problem="MIS"):
        self.G = G
        self.N = len(G)
        L =  self.N
        self.G_comp = nx.complement(G)
        self.gs_states = None
        self.gs_en = None
        self.deg = None

        dimQ = 2

        self.Id = qu.tensor([qu.qeye(dimQ)] * L)

        temp = [[qu.qeye(dimQ)] * j +
                [qu.sigmax()] +
                [qu.qeye(dimQ)] * (L - j - 1)
                for j in range(L)
                ]

        self.X = [qu.tensor(temp[j]) for j in range(L)]

        temp = [[qu.qeye(dimQ)] * j +
                [qu.sigmay()] +
                [qu.qeye(dimQ)] * (L - j - 1)
                for j in range(L)
                ]
        self.Y = [qu.tensor(temp[j]) for j in range(L)]

        temp = [[qu.qeye(dimQ)] * j +
                [qu.sigmaz()] +
                [qu.qeye(dimQ)] * (L - j - 1)
                for j in range(L)
                ]
        self.Z = [qu.tensor(temp[j]) for j in range(L)]

        self.Had = [(self.X[j] + self.Z[j]) / np.sqrt(2) for j in range(L)]

        H_c, gs_states, gs_en, deg = self.hamiltonian_cost(problem=problem, penalty=DEFAULT_PARAMS["penalty"])

        self.H_c = H_c
        self.gs_states = gs_states
        self.gs_en = gs_en
        self.deg = deg

        self.gs_binary = self.binary_gs(gs_states)


    def binary_gs(self, gs_states):
        gs_binary = []
        for gs in gs_states:
            pos = np.where(np.real(gs.full()))[0][0]
            gs_binary.append(np.binary_repr(pos, width=self.N))
        return gs_binary


    def Rx(self, qubit_n, alpha):
        op = np.cos(alpha) * self.Id -1j * np.sin(alpha) * self.X[qubit_n]
        return op


    def Rz(self, qubit_n, alpha):
        op = np.cos(alpha) * self.Id -1j * np.sin(alpha) * self.Z[qubit_n]
        return qu.tensor(temp)


    def Rzz(self, qubit_n, qubit_m, alpha):
        op = (np.cos(alpha) * self.Id
              -1j * np.sin(alpha) * self.Z[qubit_n] * self.Z[qubit_m])
        return op


    def Rxx(self, qubit_n, qubit_m, alpha):
        op = (np.cos(alpha) * self.Id
              -1j * np.sin(alpha) * self.X[qubit_n] * self.X[qubit_m])
        return op


    def U_mix(self, beta):
        U = self.Id
        for qubit_n in range(self.N):
            U = self.Rx(qubit_n, beta) * U
        return U


    def U_c(self, gamma):
        # evolution operator of U_c
        eigen_energies = np.diagonal(self.H_c)
        evol_op = []
        for j_state, el in enumerate(eigen_energies):
            bin_state = np.binary_repr(j_state, self.N)
            eigen_state =  qu.tensor([qu.basis(2, int(e_state)) for e_state in bin_state])
            evol_op.append(np.exp(-1j * el * gamma) * eigen_state * eigen_state.dag())
        U = sum(evol_op)

        return U


    def gibbs_obj_func(self, eta):
        # Gibbs objective function
        eigen_energies = np.diagonal(self.H_c)
        evol_op = []
        for j_state, el in enumerate(eigen_energies):
            bin_state = np.binary_repr(j_state, self.N)
            eigen_state =  qu.tensor([qu.basis(2, int(e_state)) for e_state in bin_state])
            evol_op.append(np.exp(-1 * eta * el) * eigen_state * eigen_state.dag())
        gibbs = sum(evol_op)

        return gibbs



    def s2z(self, configuration):
        return [1 - 2 * s for s in configuration]


    def z2s(self, configuration):
        return [(1-z)/2 for z in configuration]


    def str2list(self, s):
        list_conf = []
        skip = False
        for x in s:
            if skip:
                skip = False
                continue
            if x == "-":
                list_conf.append(-1)
                skip = True
            if x != "-":
                list_conf.append(int(x))
        return list_conf


    def hamiltonian_cost(self, problem, penalty):
        if problem == "MIS":
            H_0 = [-1*self.Z[i] / 2 for i in range(self.N)]
            H_int = [(self.Z[i] * self.Z[j] - self.Z[i] - self.Z[j]) / 4 for i, j in  self.G.edges]
            ## Hamiltonian_cost is minimized by qaoa so we need to consider -H_0
            # in order to have a solution labeled by a string of 1s
            H_c = -sum(H_0) + penalty * sum(H_int)
            energies, eigenstates = H_c.eigenstates(sort = 'low')

            degeneracy = next((i for i, x in enumerate(np.diff(energies)) if x), 1) + 1
            deg = (degeneracy > 1)
            gs_en = energies[0]
            gs_states = [state_gs for state_gs in eigenstates[:degeneracy]]

        elif problem == "MAX-CUT":
            H_int = [(self.Id - self.Z[i] * self.Z[j]) / 2 for i, j in  self.G.edges]
            H_c = -1 * sum(H_int)
            energies, eigenstates = H_c.eigenstates(sort = 'low')

            degeneracy = next((i for i, x in enumerate(np.diff(energies)) if x), 1) + 1
            deg = (degeneracy > 1)
            gs_en = energies[0]
            gs_states = [state_gs for state_gs in eigenstates[:degeneracy]]

        else:
            print("problem sohuld be one of the following: MIS, MAX-CUT")
            exit(-1)
        return H_c, gs_states, gs_en, deg


    def evaluate_cost(self, configuration):
        '''
        configuration: strings of 0,1. The solution (minimum of H_c) is labelled by 1
        '''
        qstate_from_configuration = qu.tensor([qu.basis(2, _) for _ in configuration])
        cost = qu.expect(self.H_c, qstate_from_configuration)

        return cost


    def quantum_algorithm(self,
                          params,
                          obj_func="energy",
                          penalty=DEFAULT_PARAMS["penalty"]):

        depth = int(len(params)/2)
        gammas = params[::2]
        betas = params[1::2]
        state_0 = qu.tensor([qu.basis(2, 0)] * self.N)

        for h in self.Had:
            state_0 = h * state_0

        for p in range(depth):
            state_0 = self.U_mix(betas[p]) * self.U_c(gammas[p]) * state_0

        mean_energy = qu.expect(self.H_c, state_0)

        variance = qu.expect(self.H_c * self.H_c, state_0) - mean_energy**2

        fidelities= []
        for gs_state in self.gs_states:
            fidelities.append(np.abs(state_0.overlap(gs_state))**2)

        fidelity_tot = np.sum(fidelities)

        if obj_func == "energy":
            return state_0, mean_energy, variance, fidelity_tot

        elif obj_func == "gibbs":
            gibbs_op = self.gibbs_obj_func(eta=2)
            mean_gibbs = -np.log(qu.expect(gibbs_op, state_0))
            return state_0, mean_gibbs, variance, fidelity_tot

    def generate_random_points(self,
                               N_points,
                               depth,
                               angles_bounds,
                               fixed_params=None):
#         print(extrem_params)
        np.random.seed(DEFAULT_PARAMS['seed'])
        random.seed(DEFAULT_PARAMS['seed'])

#         for i in range(N_points):
#             if fixed_params is None:
#                 x = [random.uniform(extrem_params[0], extrem_params[1]) for _ in range(depth)]
#                 x = [random.uniform(extrem_params[0], extrem_params[1]) for _ in range(depth)]
#             else:
#                 x = fixed_params + [random.uniform(extrem_params[0], extrem_params[1]) for _ in range(2)]
#             X.append(x)
#             state_0, mean_energy, variance, fidelity_tot = self.quantum_algorithm(x)
#             Y.append(mean_energy)

        hypercube_sampler = qmc.LatinHypercube(d=depth*2, seed = DEFAULT_PARAMS['seed'])
        X = hypercube_sampler.random(N_points)
        l_bounds = np.repeat(angles_bounds[:,0], depth)
        u_bounds = np.repeat(angles_bounds[:,1], depth)
        X = qmc.scale(X, l_bounds, u_bounds).astype(int)
        X = X.tolist()
        Y = []
        for x in X:
            state_0, mean_energy, variance, fidelity_tot = self.quantum_algorithm(x)
            Y.append(mean_energy)

#            y, var_y, fid_sampled, fid_exact, sol_ratio, _ , _ = self.apply_qaoa(x)
#            Y.append(y)
#            data_train.append([var_y, fid_sampled, fid_exact, sol_ratio])


        return X, Y
