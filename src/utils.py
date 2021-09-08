import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import multi_dot
import itertools
import functools
from tqdm import tqdm
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

def classical_solution(G, hamiltonian, omega = 0, verbose = 0):    
    '''
    Runs on all 2^n configurations of the graph to find the maxclique(s)

    Parameters
    ----------
    G : nx graph
        the graph to estimate the max_clique
    hamiltonian : string
        Decide which hamiltonian cost is defined on the graph
    verbose : int
        What to print
        0: nothing
        1: print info
        2: plot graph
        The default is 0.

    Returns
    -------
    d : dict of floats.
        the maxclique(s)

    '''
    
    #Evaluate for every possible configuration
    lst = list(itertools.product([0, 1], repeat=len(G)))
    results = {}
    for i in tqdm(range(2**len(G))):
        if hamiltonian == 'maxclique':
            results[lst[i]] = max_clique_cost(G, omega, np.array(lst[i]))
        if hamiltonian == 'maxcut':
            results[lst[i]] = max_cut_cost(G, np.array(lst[i]))
    
    sol = np.unique(list(results.values()), return_counts = True)
    if verbose>0:
        print('All possible solutions: \n')
        print(sol[0])
        print(sol[1])
        
    d = dict((k, v) for k, v in results.items() if v == np.min(list(results.values())))
    
    if verbose>0:
        print('There are {} MAXCLIQUE(S) with energy: \n'.format(len(d)), d)
    
    if verbose>1:
        fig = plt.figure(figsize=(4, 4))
        val, counts = np.unique(list(results.values()), return_counts = True)
        plt.bar(val, counts)
        plt.xlabel('Energy')
        plt.ylabel('Counts')
        plt.title('Statistics of solutions')
    
        #PLot one of the largest cliques
        fig = plt.figure(figsize = (2,2))
        plt.title('MaxClique')
        colors       = list(d.keys())[0]
        pos          = nx.circular_layout(G)
        nx.draw_networkx(G, node_color=colors, node_size=200, alpha=1, pos=pos)
    
    return d
    
