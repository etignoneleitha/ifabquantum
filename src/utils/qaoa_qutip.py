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

from typing import List, Tuple, Union



class qaoa_qutip(object):

    def __init__(self, 
                    G: nx.networkx, 
                    shots: Union[str, float] = None, 
                    problem: str = "MIS", 
                    gate_noise: Union[str, float] = None) -> None:
        
        """Initializes a class that simulates a qaoa circuit to solve a problem
        on a graph with qutip.
        
        Args:
            G: graph
            shots: number of shots to estimate the energy of qaoa
            problem: name of the problem to solve
            gate_noise: amount of noise to add to each gate
            
        """
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


    def binary_gs(self, gs_states: List[qu.Qobj]) -> List[str]:
        """Convert Qobj ground state into a bitstring.
        
        Args:
            gs_states : groundstates.
        
        Returns:
            Groundstates written as bitstrings.
        """
        
        gs_binary = []
        
        for gs in gs_states:
            pos = np.where(np.real(gs.full()))[0][0]
            gs_binary.append(np.binary_repr(pos, width=self.N))
        
        return gs_binary


    def Rx(self, qubit_n: int, alpha: float) -> qu.Qobj:
        """Creates X-rotation operator.
        
        Args:
            qubit_n : position of the qubit.
            alpha: angle of rotation.
        
        Returns:
            Rx(alpha) acting on qubit_n.
        """
        op = np.cos(alpha) * self.Id -1j * np.sin(alpha) * self.X[qubit_n]
        
        return op


    def Rz(self, qubit_n: int, alpha: float) -> qu.Qobj:
        """Creates Z-rotation operator.
        
        Args:
            qubit_n : position of the qubit.
            alpha: angle of rotation.
        
        Returns:
            Rz(alpha) acting on qubit_n.
        """
        
        op = np.cos(alpha) * self.Id -1j * np.sin(alpha) * self.Z[qubit_n]
        return op


    def Rzz(self, qubit_n: int, qubit_m: int, alpha: float) -> qu.Qobj:
        """Creates Rzz-rotation operator.
        
        Args:
            qubit_n : position of the first qubit.
            qubit_m : position of the second qubit
            alpha: angle of rotation.
        
        Returns:
            Rzz(alpha) acting on qubit_n and qubit_m.
        """
        
        op = (np.cos(alpha) * self.Id
              -1j * np.sin(alpha) * self.Z[qubit_n] * self.Z[qubit_m])
        
        return op


    def Rxx(self, qubit_n: int, qubit_m: int, alpha: float) -> qu.Qobj:
        """Creates Rxx-rotation operator.
        
        Args:
            qubit_n : position of the first qubit.
            qubit_m : position of the second qubit
            alpha: angle of rotation.
        
        Returns:
            Rxx(alpha) acting on qubit_n and qubit_m.
        """
        
        op = (np.cos(alpha) * self.Id
              -1j * np.sin(alpha) * self.X[qubit_n] * self.X[qubit_m])
        return op


    def U_mix(self, beta: float) -> qu.Qobj:
        """Creates the mixing operator.
        
        Args:
            beta: the angle of rotation.
            
        Returns:
            X(beta) applied to all qubits.
        """
        
        U = self.Id
        for qubit_n in range(self.N):
            U = self.Rx(qubit_n, beta) * U
        return U


    def U_c(self, gamma: float) -> qu.Qobj:
        """Creates the evolution operator.
        
        Args:
            gamma: the angle of rotation.
            
        Returns:
            Evolution operator of the system under the problem hamiltonian .
        """
                    
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

    def noisy_hamiltonian(self, gate_noise: float) -> qu.Qobj:
        """Creates problem hamiltonian with noise.
        
           Gaussian noise centered at 1 with variance gate_noise 
           is added on each operator.
        
        Args:
            gate_noise: noise variance.
            
        Returns:
            Hamiltonian operator of the problem with noise.
        
        """
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


    def s2z(self, configuration: str) -> str:
        """"Converts sequences of 0's and 1's into +1's and -1's.
        
        Args:
            configuration: the sequence of 0 and 1.
        
        Returns:
            Sequence where each 0 is a +1 and each 1 is a -1.
        """
        
        return [1 - 2 * int(s) for s in configuration]


    def z2s(self, configuration: str) -> str:
        """"Converts sequences of +1's and -1's into 0's and -1's.
        
        Args:
            configuration: the sequence of +1 and -1.
        
        Returns:
            Sequence where each +1 is a 0 and each -1 is a 1.
        """
        
        return [(1-z)/2 for z in configuration]

        
    def hamiltonian_cost(self, 
                         problem: str, 
                         penalty: float) -> Tuple[qu.Qobj,
                                                  List[qu.Qobj], 
                                                  float,
                                                  int,
                                                  np.ndarray, 
                                                  np.ndarray,
                                                  List[qu.Qobj]]:
                                                                      
        """Creates the problem hamiltonian.
        
        Args:
            problem: name between MIS, MAXCUT, ISING, ISING_TRANSV.
            penalty: the penalty value for the MIS hamiltonian.
        
        Returns
            Tuple of operator hamiltonian, its groundstate, the energy of
            the groundstate, its degeneracy, all the eigenstates, their 
            energies and the projectors built with all the eigenstates.
        """
        
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

    def classical_cost(self, bitstring: str, 
                        penalty : float =DEFAULT_PARAMS["penalty"]) -> float:
        """ Calculates the classical cost of a bit string depending on the problem.
        
        Args:
            bitstring : string of bits
            penalty: penalt of the problem (if it is MIS)
            
        Returns:
            classical cost
        """
    
              
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
                          params: np.ndarray,
                          penalty: float = DEFAULT_PARAMS["penalty"]
                          ) -> Tuple[qu.Qobj, float, float, float]:
                          
        """Runs the QAOA algorithm with given parameters.
        
        Args:
            params: gamma and beta parameters rotation of the gates
            penalty: penalty for the MIS problem
            
        Returns:
            The state obtained at the end of the QAOA circuit with its
            energy, its variance and its fidelity
        """
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
            
            
            return state_0, mean_energy, variance, fidelity_tot

            
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
                               N_points: int,
                               depth: int,
                               angles_bounds: Tuple
                               ) -> Tuple[np.ndarray, np.ndarray]:
        
        """Generates N_points random point distributed in the parameter space
        defined by the angle_bounds with a latin hypercube sampler.
        
        Args:
            N_points: number of points to generate.
            depth: depth of the QAOA circuit.
            angles_bounds: bounds of the points to generate.
            
        Returns:
            Points with their associated QAOA energy.
        """
         
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
            
        return X, Y
        
    def get_landscape(self, 
                      angle_bounds: Tuple, 
                      num: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """Runs QAOA at p=1 on a grid (num, num) in the limits given by 
        angle_bounds.
        
        Args:
            angle_bounds: bounds of the grid.
            num: refinement of the grid.
        
        Returns:
            Energies, fidelities and variances at each combination of angles
            of the grid.
        
        """
        
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
        
        return energies, fidelities, variances
        
    def classical_solution(self, save: bool = False) -> Tuple[dict, float, dict]:
        """Runs through all 2^n possible configurations and returns the 
           classical solution of the problem.
        
        Args:
            save: decide to save to file the information
        Returns: 
            A dictionary with the energy of each solution bitstring, the energy
            of the solution a dictionary of the first excited states with their
            energies. 
        """
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
        
