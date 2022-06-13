import networkx as nx
from collections import Counter, defaultdict, namedtuple
from itertools import product
import pandas as pd
import random
import numpy as np
from pathlib import Path


# QUANTUM
#from qiskit import Aer, QuantumCircuit, execute
import qutip as qu
from  utils.default_params import *

# VIZ
from matplotlib import pyplot as plt
from utils.default_params import *

from scipy.stats import qmc

import openfermion
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermionpyscf import run_pyscf


class qaoa_qutip(object):

    def __init__(self, G, 
                    shots = None, 
                    problem="MIS", 
                    gate_noise = None,
                    bond_distance = 0.7414):
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

        H_c, gs_states, gs_en, deg, eigenstates, eigenvalues= self.hamiltonian_cost(
                                                                problem=problem, 
                                                                penalty=DEFAULT_PARAMS["penalty"],
                                                                bond_distance =  bond_distance
                                                                )

        self.H_c = H_c
        self.gs_states = gs_states
        self.gs_en = gs_en
        self.deg = deg
        self.eigenstates = eigenstates
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


    # def U_c(self, gamma):
#         # evolution operator of U_c
#         eigen_energies = np.diagonal(self.H_c)
#         evol_op = []
#         for j_state, el in enumerate(eigen_energies):
#             bin_state = np.binary_repr(j_state, self.N)
#             eigen_state =  qu.tensor([qu.basis(2, int(e_state)) for e_state in bin_state])
#             evol_op.append(np.exp(-1j * el * gamma) * eigen_state * eigen_state.dag())
#         U = sum(evol_op)
# 
#         return U

   #  def U_c(self, gamma):
#         # evolution operator of U_c
#         evol_op = []
#         for energy, eigen_state in zip(self.energies, self.eigenstates):
#             evol_op.append(np.exp(-1j * energy * gamma) * eigen_state * eigen_state.dag())
#             
#         U = sum(evol_op)
# 
#         return U

    def U_c(self, gamma):
        # evolution operator of U_c
        if self.gate_noise == None:
            eigen_energies,eigen_states = np.linalg.eig(self.H_c)
        else:
            self.H_simulation = self.noisy_hamiltonian(self.gate_noise)
            eigen_energies, eigen_states = np.linalg.eig(self.H_simulation)
            
        evol_op = []
       #  for j_state, el in enumerate(eigen_energies):
#             bin_state = np.binary_repr(j_state, self.N)
#             eigen_state =  qu.tensor([qu.basis(2, int(e_state)) for e_state in bin_state])
#             evol_op.append(np.exp(-1j * el * gamma) * eigen_state * eigen_state.dag())
#         
        for e_en, e_state in zip(eigen_energies, eigen_states):
            state_ = qu.Qobj(e_state, dims = [[2]*self.N] +[ [1]*self.N])
            proj = state_.proj()
            evol_op.append(np.exp(-1j * e_en * gamma) * proj)
        
        U= sum(evol_op)

        return U

    def noisy_hamiltonian(self, gate_noise):
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
            
        if self.problem == "MAX-CUT":
            link_noise = np.random.normal(1, 
                                          gate_noise, 
                                          len(self.G.edges))
            H_int = [link_noise[k]*(self.Id - self.Z[i] * self.Z[j]) / 2 for k, (i,j) in  enumerate(self.G.edges)]
            H_c = -1 * sum(H_int)
            
        
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


    def create_qubit_hamiltonian_h2(self, bond_distance, transform = 'JW'):
        '''
        creates the qubit hamiltonian for the H2 problem with a given bond
        distance and using JW transform (default) or BK (bravyi_kitaev)
        and saves it as a .data file
        '''
        
        bond_distance = bond_distance
        geometry = [ [ 'H' , [ 0 , 0 , 0 ] ] ,
                       [ 'H' , [ 0 , 0 , bond_distance] ] ]
        basis= 'sto3g'
        multiplicity = 1
        charge = 0
        description = str(bond_distance)
        
        #create a molecule object with only general information, still no
        #one-body or two-body integral have been calculated (they would return
        #None now)
        molecule= MolecularData(geometry , basis, multiplicity, charge , description)
        print('Molecule has automatically generated name {}'.format(
            molecule.name))
        print('Information about this molecule would be saved at:\n{}\n'.format(
            molecule.filename))
        print('This molecule has {} atoms and {} electrons.'.format(
            molecule.n_atoms, molecule.n_electrons))
        for atom, atomic_number in zip(molecule.atoms, molecule.protons):
            print('Contains {} atom, which has {} protons.'.format(
                atom, atomic_number))   
        
        #Now run a backend eletronic structure package to obtain the integrals
        # and all the information and energies
        h2_molecule = run_pyscf(molecule,
                                run_mp2 = True,
                                run_cisd = True,
                                run_ccsd = True,
                                run_fci = True)
        h2_filename = h2_molecule.filename
        h2_molecule.save()  
        
        print('\nAt bond length of {} angstrom, molecular hydrogen has:'.format(
            bond_distance))
        print('Hartree-Fock energy of {} Hartree.'.format(molecule.hf_energy))
        print('MP2 energy of {} Hartree.'.format(molecule.mp2_energy))
        print('FCI energy of {} Hartree.'.format(molecule.fci_energy))
        print('Nuclear repulsion energy between protons is {} Hartree.'.format(
            molecule.nuclear_repulsion))
            
       #  for orbital in range(molecule.n_orbitals):
#             print('Spatial orbital {} has energy of {} Hartree.'.format(
#                 orbital, molecule.orbital_energies[orbital]))
        
        
        one_body_integrals = h2_molecule.one_body_integrals
        two_body_integrals = h2_molecule.two_body_integrals
        print('Integrals\nOnebody:\n', one_body_integrals)
        print('\nTwo body:',two_body_integrals)
        
        
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        
        #The hamiltonian is written in 2nd qntzation where (i,j) means acting on 
        #site i with fermion operator j and j can be either 0 (destroy operator)
        #or 1 (creation operator a^\dagger)
        print('Hamiltonian:\n',molecular_hamiltonian)
        
        
        #Map operator to fermions and qubits.
        fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
        
        if transform == 'JW':
            qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
        
        if transform == 'BK':
            qubit_hamiltonian = bravyi_kitaev(fermion_hamiltonian)
        qubit_hamiltonian.compress() #only removes zero-entries
        
        output_folder = str(Path(__file__).parents[2] / "output")
                
        openfermion.utils.save_operator(
            qubit_hamiltonian,
            file_name=f'H2_{transform}_hamiltonian_bound_distance_{bond_distance}',
            data_directory=output_folder,
            allow_overwrite=True,
            plain_text=True
        )
        
    def load_qubit_hamiltonian(self, bond_distance,  transform = 'BK'):
        ''' 
        Loads a jordan wigener transformed qubit hamiltonian created
        with the method create_qubit_hamiltonian and extracts the terms
        and the gates to be applied
        '''
        directory_name = str(Path(__file__).parents[2] / "output")
        file_name = f'H2_{transform}_hamiltonian_bound_distance_{bond_distance}.data'
        
        with open(directory_name + '/' + file_name) as f:
            a = f.read()
        
        #erase the useless parts
        a = a.replace('[', '')
        a = a.replace(']', '')
        a = a.replace(' +', '')
        
        #splits into lines and erases the first one 
        
        split_into_lines = a.split('\n')[1:]
        
        ham = 0
        
        #run through each line extracting the terms
        for new_line in split_into_lines:
            splitted_line = new_line.split(' ')
            
            
            #first element is the coefficient of the hamiltonian term
            #second element is a list of pauli gates
            coefficient = float(splitted_line[0])
            operators_ = splitted_line[1:]
            
            hamiltonian_term = coefficient * self.Id
            
            #run through each pauli gate XO Y1 ... of this line and adds 
            #it to the hamiltonian term
            for operator_ in operators_:
                
                if operator_ == '':
                    break
                    
                pauli_gate = operator_[0]
                qubit = int(operator_[1])
                
                if pauli_gate == 'X':
                    hamiltonian_term = hamiltonian_term * self.X[qubit]
                
                if pauli_gate == 'Y':
                    hamiltonian_term *= self.Y[qubit]
                
                if pauli_gate == 'Z':
                    hamiltonian_term *= self.Z[qubit]
                    
            ham += hamiltonian_term
            
        return ham
                
        
        
    def hamiltonian_cost(self, problem, penalty, bond_distance = 0.7414):
        '''
        receiving also bond length for the case of the H2 molecule
        starndard value is the lowest energy
        '''
        
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
        
        elif problem == 'H2':
            self.shots = None
            if self.N != 4:
                print('WARNING\n you are running this with the incorrect'
                        f'number of nodes, shoule be 4 but you set {self.N}')
                
            self.create_qubit_hamiltonian_h2(bond_distance, transform = 'JW')
            H_c = self.load_qubit_hamiltonian(bond_distance, transform = 'JW')
        
            
            
           # #proj = 0.25*self.Id -0.25*self.Z[0]*self.Z[1] \
#                 #-0.25*self.Z[2]*self.Z[3] + 0.25*self.Z[0]*self.Z[1]*self.Z[2]*self.Z[3]
#             proj = 0.25*self.Id -0.25*self.Z[0]*self.Z[2] \
#                 -0.25*self.Z[1]*self.Z[3] + 0.25*self.Z[0]*self.Z[1]*self.Z[2]*self.Z[3]
#                 
#             H_red = proj.conj() * H_c * proj
#             print(H_red)
#             H_red_shift = qu.Qobj(np.real(H_red[4:12,4:12]), dims = [[2,2, 2],[2,2,2]])
#             print(H_red_shift)
#             
#             #now have to redefine operators acting on a L-1 qubit space
#             dimQ = 2
#             L = 3
#             Id_3 = qu.tensor([qu.qeye(dimQ)] * L)
# 
#             temp = [[qu.qeye(dimQ)] * j +
#                     [qu.sigmax()] +
#                     [qu.qeye(dimQ)] * (L - j - 1)
#                     for j in range(L)
#                     ]
# 
#             X_3 = [qu.tensor(temp[j]) for j in range(L)]
# 
#             temp = [[qu.qeye(dimQ)] * j +
#                     [qu.sigmay()] +
#                     [qu.qeye(dimQ)] * (L - j - 1)
#                     for j in range(L)
#                     ]
#             Y_3 = [qu.tensor(temp[j]) for j in range(L)]
# 
#             temp = [[qu.qeye(dimQ)] * j +
#                     [qu.sigmaz()] +
#                     [qu.qeye(dimQ)] * (L - j - 1)
#                     for j in range(L)
#                     ]
#             Z_3 = [qu.tensor(temp[j]) for j in range(L)]
#             
#             
#             reorder = 0.5*(Id_3 + Z_3[0]*Z_3[2] \
#                          - Z_3[0]*X_3[1]*Z_3[2] + X_3[1])
#             print(reorder)
#             H_red_shift_reordered = reorder.conj() * H_red_shift * reorder
#             print(H_red_shift_reordered)
#             
#             H_red_shift_reordered_reshifted = qu.Qobj(np.real(H_red_shift_reordered[2:6, 2:6]),
#                                                         dims = [[2,2],[2,2]])
#             print(H_red_shift_reordered_reshifted)
#             
#             eig, eigsta = H_red_shift_reordered_reshifted.eigenstates(sort = 'low')
#             print(eig, eigsta)
            
            
        elif problem == 'H2_reduced':
            '''following notation from eq(1) O'Malley Scalable Quantum 
               Simulation of Molecular Energies'''
            g_0=-0.1927
            g_1=0.2048
            g_2=-0.0929
            g_3=0.4588
            g_4=0.1116
            g_5=0.1116 
            
            H_one = g_0 * self.Id + g_1 * self.Z[0] + g_2 * self.Z[1]
            
            H_two = g_3 * self.Z[0] * self.Z[1] \
                    + g_4 * self.Y[0] * self.Y[1] \
                    + g_5 * self.X[0] * self.X[1] 
                    
            H_c = H_one + H_two
            
            
        elif problem == 'H2_BK_reduced':
        
            h=-0.81261
            h_0=0.171201
            h_1=0.16862325
            h_2=-0.2227965
            h_10=0.171201
            h_20=0.12054625
            h_31=0.17434925
            h_xzx = 0.04532175
            h_yzy = 0.04532175
            h_210 = 0.165868
            h_320 = 0.12054625
            h_321 = -0.2227965
            h_zxzx = 0.04532175
            h_zyzy = 0.04532175
            h_3210 = 0.165868
            
            g_0 = h - h_1 - h_31
            g_1 = h_0 - h_10
            g_2 = h_2 - h_321
            g_3 = h_20 - h_210 + h_320 - h_3210
            g_4 = (-1)*h_xzx - h_zxzx
            g_5 = (-1)*h_yzy - h_zyzy
            
            H_one = g_0 * self.Id + g_1 * self.Z[0] + g_2 * self.Z[1]
            
            H_two = g_3 * self.Z[0] * self.Z[1] \
                    + g_4 * self.X[0] * self.X[1] \
                    + g_5 * self.Y[0] * self.Y[1] 
                    
            H_c = H_one + H_two
            
        
        else:
            print("problem sohuld be one of the following: MIS, MAXCUT")
            exit(-1)
            
        energies, eigenstates = H_c.eigenstates(sort = 'low')
        degeneracy = next((i for i, x in enumerate(np.diff(energies)) if x), 1) + 1
        gs_en = energies[0]
        
        # matrix = np.column_stack([eigenstates[i].full() for i in range(len(eigenstates))])
#         matrix_2 = np.zeros((16, 17))
#         matrix_2[:,0] = energies
#         matrix_2[:,1 :] = matrix.T
#         col_names = ['energies'] + [str(np.binary_repr(i ,4)) for i in range(16)]
#         df = pd.DataFrame(matrix_2, columns = col_names)
#         pd.set_option('display.precision', 3)
#         df.save_to_csv()
#         exit()
        gs_states = [state_gs for state_gs in eigenstates[:degeneracy]]
            
        return H_c, gs_states, gs_en, degeneracy, eigenstates, energies

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
            #print(results_sorted)
            
            # 
#             sol_ratio = 0
#             if len(results_sorted)>1 :
#                 for i in range(len(results_sorted)):
#                     first_key, second_key =  list(results_sorted.keys())[i:i+2]
#                     
#                     if (first_key in self.gs_binary):
#                         if (second_key in self.gs_binary):
#                             continue
#                         elif(results_sorted[second_key] > 0):
#                             sol_ratio = results_sorted[first_key]/results_sorted[second_key]
#                         else:
#                             sol_ratio = self.shots
#             else:
#                 first_key =  list(result_sorted.keys())[0]
#                 if (first_key in self.gs_binary):
#                     sol_ratio = self.shots    #if the only sampled value is the solution the ratio is infinte so we put the # of shots
#             
#             return sol_ratio
                
            key_sorted = list(results_sorted.keys())
            mean_energy = sum([self.classical_cost(s)*results_sorted[s] for s in results_sorted.keys()])/self.shots
            #variance = sum([
            #    ((self.classical_cost(s) - mean_energy)*list(results_sorted.values())[s])**2 
            #                    for s in key_sorted])/self.shots
            variance = 0
            
            
            fidelities= []
            for gs_state in self.gs_binary:
                try:
                    fidelities.append(results_sorted[gs_state])
                except:
                    fidelities.append(results_sorted[gs_state])
                

            fidelity_tot = np.sum(fidelities)/self.shots
            
            return state_0, mean_energy, variance, fidelity_tot, 
            

    def generate_random_points(self,
                               N_points,
                               depth,
                               angles_bounds,
                               fixed_params=None):
                               
        np.random.seed(DEFAULT_PARAMS['seed'])
        random.seed(DEFAULT_PARAMS['seed'])
        X = []
        Y = []
        # for i in range(N_points):
#             x = np.random.uniform(angles_bounds[0], angles_bounds[1], depth*2)
#             X.append(x.tolist())
#             state_0, mean_energy, variance, fidelity_tot = self.quantum_algorithm(x)
#             Y.append(mean_energy)

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
        
    def get_landscape(self, angle_bounds, num):
    
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
        
    def classical_solution(self, save = False):
        '''
        Runs through all 2^n possible configurations and estimates the solution
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
        print('First excited states:', {k: results[k] for k in list(results)[1:5]})
        print('Energy distribution')
        print(df)
        if save:
            df.to_csv('output/energy_statistics.dat')
        
        
        return d, en
        
