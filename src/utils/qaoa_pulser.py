import numpy as np
import igraph
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2
from pulser.simulation import SimConfig
from itertools import product
from utils.default_params import *
import random


from scipy.optimize import minimize
from qutip import *

class qaoa_pulser(object):

	def __init__(self, pos, noise = False):
		self.omega = 1.
		self.delta = 1.
		self.U = 10
		self.pos_to_graph(pos)
		self.Nqubit = len(self.G)
		self.G_comp = nx.complement(self.G)
		self.gs_state = None
		self.gs_en = None
		self.deg = None
		self.qubits_dict = dict(enumerate(pos))
		self.reg = Register(self.qubits_dict)
		self.quantum_noise = noise

	def pos_to_graph(self, pos): #d is the rbr
		d = Chadoq2.rydberg_blockade_radius(self.omega)
		self.G = nx.Graph()
		edges=[]
		distances = []
		for n in range(len(pos)-1):
			for m in range(n+1, len(pos)):
				pwd = ((pos[m][0]-pos[n][0])**2+(pos[m][1]-pos[n][1])**2)**0.5
				distances.append(pwd)
				if pwd < d:
					edges.append([n,m]) # Below rbr, vertices are connected
		self.G.add_nodes_from(range(len(pos)))
		self.G.add_edges_from(edges)

	def create_quantum_circuit(self, param, time  = 3000):
		seq = Sequence(self.reg, Chadoq2)
		seq.declare_channel('ch0','rydberg_global')
		middle = int(len(param)/2)
		param = np.array(param)*1 #wrapper
		t = param[:middle] #associated to H_c
		tau = param[middle:] #associated to H_0
		p = len(t)
		for i in range(p):
			ttau = int(tau[i]) - int(tau[i]) % 4
			tt = int(t[i]) - int(t[i]) % 4
			pulse_1 = Pulse.ConstantPulse(ttau, self.omega, 0, 0) # H_M
			pulse_2 = Pulse.ConstantPulse(tt, self.delta, 1, 0) # H_M + H_c
			seq.add(pulse_1, 'ch0')
			seq.add(pulse_2, 'ch0')
		seq.measure('ground-rydberg')
		simul = Simulation(seq, sampling_rate=0.1)
	
		return simul
	
	def quantum_loop(self, param):
		sim = self.create_quantum_circuit(param)
		if self.quantum_noise:
			cfg = SimConfig(noise=('SPAM', 'dephasing', 'doppler'))
			sim.add_config(cfg)
		results = sim.run()
		count_dict = results.sample_final_state(N_samples=1000) #sample from the state vector
		return count_dict, results.states

	def plot_distribution(self, C):
		C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
		color_dict = {key: 'g' for key in C}
		indexes = ['01011', '00111']  # MIS indexes
		for i in indexes:
			color_dict[i] = 'red'
		plt.figure(figsize=(12,6))
		plt.xlabel("bitstrings")
		plt.ylabel("counts")
		plt.bar(C.keys(), C.values(), width=0.5, color = color_dict.values())
		plt.xticks(rotation='vertical')
		plt.show()
	
	def list_operator(self, op):
		''''
		returns a a list of tensor products with op on site 0, 1,2 ...
		'''
		op_list = []
	
		for n in range(self.Nqubit):
			op_list_i = []
			for m in range(self.Nqubit):
				op_list_i.append(qeye(2))
	   
			op_list_i[n] = op
			op_list.append(tensor(op_list_i)) 
	
		return op_list
	
	def calculate_physical_gs(self):
		'''
		returns groundstate and energy 
		'''

		## Defining lists of tensor operators 
		ni = (qeye(2) - sigmaz())/2

		sx_list = self.list_operator(sigmax())
		sz_list = self.list_operator(sigmaz())
		ni_list = self.list_operator(ni)
	
		H=0
		for n in range(self.Nqubit):
			H += self.omega * sx_list[n]
			H -= self.delta * ni_list[n]
		for i, edge in enumerate(self.G.edges):
			H +=  self.U*ni_list[edge[0]]*ni_list[edge[1]]
		energies, eigenstates = H.eigenstates(sort = 'low')
		if energies[0] ==  energies[1]:
			print('DEGENERATE GROUND STATE')
			self.deg = True
			self.gs_en = energies[:2]
			self.gs_state = eigenstates[:2]
		else:
			self.deg = False
			self.gs_en = energies[0]
			self.gs_state = eigenstates[0]
	
		return self.gs_en, self.gs_state, self.deg

	def generate_random_points(self, Npoints, depth, extrem_params):
		X = []
		Y = []
		
		np.random.seed(DEFAULT_PARAMS['seed'])
		random.seed(DEFAULT_PARAMS['seed'])
        
		for i in range(Npoints):
			x = [np.random.randint(extrem_params[0],extrem_params[1]),
					np.random.randint(extrem_params[0],extrem_params[1])]*depth
			X.append(x)
			y = self.expected_energy(x)
			Y.append(y)
	
		if Npoints == 1:
			X = np.reshape(X, (depth*2,))
		return X, Y

	def fidelity_gs_sampled(self, x):
		'''
		Fidelity sampled means how many times the solution(s) is measured
		'''
		C = self.get_sampled_state(x)
		
		indexes = ['01011', '00111']  # MIS indexes
	
		return (C[indexes[0]]+C[indexes[1]])
		
	def fidelity_gs_exact(self, param):
		'''
		Return the fidelity of the exact qaoa state (obtained with qutip) and the 
		exact groundstate calculated with the physical hamiltonian of pulser
		'''
		C, evolution_states = self.quantum_loop(param)
		if self.gs_state == None:
			self.calculate_physical_gs()
		fid = fidelity(self.gs_state, evolution_states[-1])
		return fid

	def get_cost_dict(self, counter):
		total_cost = 0
		for key in counter.keys():
			cost = 0
			penalty = self.U
			A = np.array(adjacency_matrix(self.G).todense())
			z = np.array(tuple(key),dtype=int)
			for i in range(len(z)):
				for j in range(i,len(z)):
					cost += A[i,j]*z[i]*z[j]*penalty # if there's an edge between i,j and they are both in |1> state.

			cost -= np.sum(z) #to count for the 0s instead of the 1s
			total_cost += cost * counter[key]
		return total_cost / sum(counter.values())

	def get_sampled_state(self, params):
		C, _ = self.quantum_loop(params)
		return C
		
	def expected_energy(self, param):
		C, _ = self.quantum_loop(param)
		cost = self.get_cost_dict(C)
		
		return cost
