import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import multi_dot

import functools
import networkx as nx

def create_random_graph(n, p, seed = None):
    '''
    Generates a random graph and calculates its maximum clique. 
    Est. run time for the generation is 
    O(n+m)  where m =  pn(n−1)/2 is the expected number of edges.
    
    Parameters
    ----------
    n : INT, Number of edges
    
    p : INT, Edge probability
    
    seed : INT, Seed. The default is None.

    Returns
    -------
    An Erdos-Renyi graph
    
    '''
    G = nx.fast_gnp_random_graph(n, p, seed)
    
    return G

def create_graph_from_edges(E):
    '''
    Generates a graph from a set of edges
    
    Parameters
    ----------
    E : List
        Set of edges
        
    Returns
    -------
    The Graph made with edges
    THe number of nodes

    '''
    
    G = nx.Graph()
    G.add_edges_from(E)
    N = nx.number_of_nodes(G)
    
    return G, N

def max_clique_cost(G_comp, omega, bitstring):
    '''
    Calculates Classical Max Clique Hamiltonian Cost of bistring on graph G_comp
    after the transformation [0,1] -> [1,-1]

    Parameters
    ----------
    G_comp: Graph
        The complementary graph which is needed to calculate the cost for max clique
        
    omega: float
        The penalty for the second term of Hamiltonian
        
    bitstring : List
        A list of bit values 0,1

    Returns
    -------
    The cost of the bitstring configuration for maxclique

    '''
    var = 1 - 2*bitstring
    first_term = sum(var)
    second_term = 0
    for edge in G_comp.edges:
        second_term += omega*(var[edge[0]]*var[edge[1]] - var[edge[0]] - var[edge[1]])
    return first_term + second_term 

def max_cut_cost(G, bitstring):
    '''
    Calculates Classical Max Cut Hamiltonian Cost of bistring on graph G
    after the transformation [0,1] -> [1,-1]

    Parameters
    ----------
    G_comp: Graph
        The graph which is needed to calculate the cost for max cut
        
    bitstring : List
        A list of bit values 0,1

    Returns
    -------
    The cost of the bitstring configuration

    '''
    var = 1 - 2*bitstring
    C = 0
    for edge in G.edges:
        C +=  (1-var[edge[0]]*var[edge[1]])/2
    return C  
    
