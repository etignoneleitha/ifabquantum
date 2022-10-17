import networkx as nx
from itertools import product
import pandas as pd
import random
import numpy as np
from pathlib import Path
from scipy.stats import qmc


# QUANTUM
import qutip as qu
from  utils.default_params import *

# VIZ
from matplotlib import pyplot as plt
from utils.default_params import *



class qaoa_qutip(object):

    def __init__(self, G, 
                    shots = None, 
                    problem="MIS", 
                    gate_noise = None):
        self.shots = shots
        self.gate_noise = gate_noise
        self.problem = problem
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

        H_c, gs_states, gs_en, deg, eigenstates, eigenvalues, projectors = self.hamiltonian_cost(
                                            problem=problem, 
                                            penalty=DEFAULT_PARAMS["penalty"]
                                            )

        self.H_c = H_c
        self.gs_states = gs_states
        self.gs_en = gs_en
        self.deg = deg
        self.eigenstates = eigenstates
        self.projectors = projectors
        self.energies = eigenvalues

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
                    
        evol_op = []
        
        if self.gate_noise is not None:
            ham = self.noisy_hamiltonian(self.gate_noise)
            
            energies, states = ham.eigenstates(sort = 'low')

            for en, state_ in zip(energies, states):
                proj = state_.proj()
                evol_op.append(np.exp(-1j * en * gamma) * proj)
            
        else:
        
            for e_en, proj in zip(self.energies, self.projectors):
        
                evol_op.append(np.exp(-1j * e_en * gamma) * proj)
        
        U = sum(evol_op)

        return U

    def noisy_hamiltonian(self, gate_noise):
    
        H_simulation  = 0
        
        if self.problem == "MIS":
            H_0 = [-1*self.Z[i] / 2 for i in range(self.N)]
            H_int = [
                (self.Z[i] * self.Z[j] - self.Z[i] - self.Z[j]) / 4 
                for i, j in  self.G.edges
                ]
                
            penalty_noise_link = np.random.normal(DEFAULT_PARAMS["penalty"], 
                                            gate_noise, 
                                            len(self.G.edges))
                                            
            qubit_noise = np.random.normal(1, 
                                            gate_noise, 
                                            self.N)
            H_int_noise = [
                penalty_noise_link[k]*(self.Z[i] * self.Z[j] - self.Z[i] - self.Z[j]) / 4 
                for k, [i, j] in  enumerate(self.G.edges)
                ]
                
            H_0_noise = [-1*qubit_noise[i]*self.Z[i] / 2 for i in range(self.N)]
                
            H_simulation = -sum(H_0_noise) + sum(H_int_noise)
            
        if self.problem == "MAXCUT":
            link_noise = np.random.normal(1, 
                                          gate_noise, 
                                          len(self.G.edges))
            H_int = [link_noise[k]*(self.Id - self.Z[i] * self.Z[j]) / 2 for k, (i,j) in  enumerate(self.G.edges)]
            H_simulation = -1 * sum(H_int)
            
            
                    
        
        return H_simulation


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
        return [1 - 2 * int(s) for s in configuration]


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
            H_int = [
                (self.Z[i] * self.Z[j] - self.Z[i] - self.Z[j]) / 4 
                for i, j in  self.G.edges
                ]
                
            ## Hamiltonian_cost is minimized by qaoa so we need to consider -H_0
            # in order to have a solution labeled by a string of 1s
            H_c = -sum(H_0) + penalty * sum(H_int)

        elif problem == "MAXCUT":
            H_int = [(self.Id - self.Z[i] * self.Z[j]) / 2 for i, j in  self.G.edges]
            H_c = -1 * sum(H_int)
        
        elif problem == 'ISING':
            H_int = [((-1) * self.Z[i] * self.Z[j]) / 2 for i, j in  self.G.edges]
            H_c = sum(H_int)

        elif problem == 'ISING_TRANSV':
            H_0 = [-1*self.X[i] / 2 for i in range(self.N)]
            H_int = H_int = [( (-1)* self.Z[i] * self.Z[j]) / 2 for i, j in  self.G.edges]
            H_c = -sum(H_0) + sum(H_int)
                    
        else:
            print("problem sohuld be one of the following: MIS, MAXCUT")
            exit(-1)
            
        energies, eigenstates = H_c.eigenstates(sort = 'low')
        degeneracy = next((i for i, x in enumerate(np.diff(energies)) if x), 1) + 1
        gs_en = energies[0]
        
        gs_states = [state_gs for state_gs in eigenstates[:degeneracy]]
        projectors = [state_.proj() for state_ in eigenstates]
        
        return H_c, gs_states, gs_en, degeneracy, eigenstates, energies, projectors

    def classical_cost(self, bitstring, penalty=DEFAULT_PARAMS["penalty"]):
    
              
        spin_string = self.s2z(bitstring)
        cost = 0
        if self.problem == 'MAXCUT':
            
            for edge in self.G.edges:
                cost -= (1 - spin_string[edge[0]]*spin_string[edge[1]])/2
        
        if self.problem == 'MIS':
            cost += sum(spin_string)/2
            for edge in self.G.edges:
                cost += penalty * (spin_string[edge[0]]*spin_string[edge[1]]
                            -spin_string[edge[0]] - spin_string[edge[1]])/4
                
        return cost
        

    def quantum_algorithm(self,
                          params,
                          obj_func="energy",
                          penalty=DEFAULT_PARAMS["penalty"]):

        depth = int(len(params)/2)
        gammas = params[::2]
        betas = params[1::2]
        states = []
        state_0 = qu.tensor([qu.basis(2, 0)] * self.N)
        states.append(state_0)
        
        for h in self.Had:
            state_0 = h * state_0
        states.append(state_0)
        
        for p in range(depth):
            state_0 =  self.U_c(gammas[p]) * state_0
            states.append(state_0)
            state_0 = self.U_mix(betas[p]) * state_0
            states.append(state_0)
        
        if self.shots is None:
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
            
        else:
            prob_each_state = np.array([np.real(x.conj() * x) for x in state_0.full()]).squeeze()
            
            samples = np.random.choice(2**self.N, 
                                               size = self.shots,
                                               replace = True,
                                               p = prob_each_state
                                               )
            samples_as_bitstrings = np.vectorize(np.binary_repr)(samples, self.N)
            results = {}
            results_counts = np.unique(samples_as_bitstrings, return_counts = True)
            results = dict(zip(results_counts[0], results_counts[1]))
            results_sorted = dict(sorted(results.items(), key=lambda item: item[1], reverse = True))
                
            key_sorted = list(results_sorted.keys())
            mean_energy = sum([self.classical_cost(s)*results_sorted[s] for s in results_sorted.keys()])/self.shots

            variance = 0
            
            
            fidelities= []
            
            for gs_state in self.gs_binary:
            
                if gs_state in results_sorted.keys():
                
                    fidelities.append(results_sorted[gs_state])
                    
                else:
                    continue
                
            fidelity_tot = np.sum(fidelities)/self.shots
            
            return state_0, mean_energy, variance, fidelity_totd
            

    def generate_random_points(self,
                               N_points,
                               depth,
                               angles_bounds,
                               fixed_params=None):
                               
        np.random.seed(DEFAULT_PARAMS['seed'])
        random.seed(DEFAULT_PARAMS['seed'])
        X = []
        Y = []

        hypercube_sampler = qmc.LatinHypercube(d=depth*2, seed = DEFAULT_PARAMS['seed'])
        X = hypercube_sampler.random(N_points)
        l_bounds = np.repeat(angles_bounds[:,0], depth)
        u_bounds = np.repeat(angles_bounds[:,1], depth)
        X = qmc.scale(X, l_bounds, u_bounds)
        X = X.tolist()
        Y = []
        for x in X:
            state_0, mean_energy, variance, fidelity_tot = self.quantum_algorithm(x)
            Y.append(mean_energy)
            
        print(X, Y)

        return X, Y
        
    def get_landscape(self, angle_bounds, num, verbose = 0):
    
        fig = plt.figure()
        energies = np.zeros((num, num))
        fidelities = np.zeros((num, num))
        variances = np.zeros((num, num))
        gammas = np.linspace(angle_bounds[0, 0],angle_bounds[0,1], num)
        betas = np.linspace(angle_bounds[1, 0],angle_bounds[1,1], num)
        for i, gamma in enumerate(gammas):
            for j, beta in enumerate(betas):
                a, en, var, fid = self.quantum_algorithm([gamma, beta])
                energies[j, i] = en
                fidelities[j, i] = fid
                variances[j, i] = var
                if verbose:
                    print(f'\u03B3: {i+1}/num, \u03B2: {j+1}/num')
        
        return energies, fidelities, variances
        
    def classical_solution(self, save = False):
        '''
        Runs through all 2^n possible configurations and returns the solution
        Returns: 
            d: dictionary with {[bitstring solution] : energy}
            en: energy of the (possibly degenerate) solution
        '''
        results = {}

        string_configurations = list(product(['0','1'], repeat=len(self.G)))

        for string_configuration in  string_configurations:
            print(string_configuration)
            single_string = "".join(string_configuration)
            results[single_string] = self.classical_cost(string_configuration)
        
        d = dict((k, v) for k, v in results.items() if v == np.min(list(results.values())))
        en = list(d.values())[0]
        
        #sort the dictionary
        results = dict(sorted(results.items(), key=lambda item: item[1]))
        
        #counts the distribution of energies
        energies, counts = np.unique(list(results.values()), return_counts = True)
        df = pd.DataFrame(np.column_stack((energies, counts)), columns = ['energy', 'counts'])
        print('\n####CLASSICAL SOLUTION######\n')
        print('Lowest energy:', d)
        first_exc = {k: results[k] for k in list(results)[self.deg:5]}
        print('First excited states:', first_exc)
        print('Energy distribution')
        print(df)
        if save:
            df.to_csv('output/energy_statistics.dat')
        
        return d, en, first_exc
        
