#!/usr/bin/env python

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from utils.qaoa import grid_search

#Set global parameters
penalty = 2
shots = 1000

#Set graph instance & its complement
E = [(0,1), (1,2), (0,2), (2,3), (2,4)]
G = nx.Graph()
N = G.number_of_nodes()
G.add_nodes_from(range(N))
G.add_edges_from(E)
G_comp = nx.complement(G)

points = grid_search(G, 20, penalty=penalty)