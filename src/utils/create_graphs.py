import networkx as nx
from networkx.generators.random_graphs import random_regular_graph
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from networkx.linalg.graphmatrix import adjacency_matrix
from pathlib import Path
from typing import List

def create_random_graph(num_nodes, average_connectivity, name_plot=False):
    G = nx.Graph()

    G.add_nodes_from(range(num_nodes))
    edges_graph = np.array(list(combinations(range(num_nodes), 2)))
    s = np.where(np.random.binomial(1, average_connectivity, len(edges_graph)))[0]
    G.add_edges_from(edges_graph[s])
    if name_plot:
        graph_name_file = str(draw)
        _plot_graph(G, graph_name_file)

    return G


def create_chair_graph(name_plot=False):
    num_nodes = 6
    G = nx.Graph()
    pos = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [0,5]])
    G.add_edges_from(pos)

    if name_plot:
        graph_name_file = "chair"
        _plot_graph(G, graph_name_file)

    return G


def create_random_regular_graph(num_nodes, degree=3, seed=1, name_plot=False):
    G = random_regular_graph(degree, num_nodes, seed=seed)
    if name_plot:
        graph_name_file = str(name_plot)
        _plot_graph(G, graph_name_file)
    return G


def create_graph_usecase_insurance(n_customers: int,
                                   n_renewal_ratios_bins: int,
                                   previous_premiums: List[float],
                                   renewal_ratios_bins: List[float],
                                   acceptance_probabilities: List[List[float]]):
    
    '''
    creates a graph for the portfolio renewal usecase of Leithà
    '''

    if type(previous_premiums) == list:
        previous_premiums = np.array(previous_premiums)
    if type(renewal_ratios_bins) == list:
        renewal_ratios_bins = np.array(renewal_ratios_bins)
    if type(acceptance_probabilities) == list:
        acceptance_probabilities = np.array(acceptance_probabilities)

    # check data dimensions
    assert len(acceptance_probabilities) == n_customers; f'there should {n_customers*n_renewal_ratios_bins} acceptance_probabilities'
    assert len(acceptance_probabilities[0]) == n_renewal_ratios_bins; f'there should {n_customers*n_renewal_ratios_bins} acceptance_probabilities'
    assert np.all(acceptance_probabilities.sum(axis=1)); 'acceptance_probabilities are not summing to 1 for at least one customer'
    
    penalty_terms = 1 + np.array(previous_premiums)*np.sum(acceptance_probabilities*previous_premiums.reshape((3, 1)),axis=1)
    
    # build unweighted graph  
    G = nx.Graph()
    for c in range(n_customers):
        tmp_G = nx.complete_graph(n_renewal_ratios_bins)
        tmp_G = nx.relabel_nodes(tmp_G, {n: (c,n) for n in tmp_G.nodes})
        G = nx.compose(G,tmp_G)

    # add node weights
    for n,d in G.nodes(data=True):
        i,j = n
        d['weight'] = acceptance_probabilities[i][j] * (1 + previous_premiums[i] * renewal_ratios_bins[j]) - (n_renewal_ratios_bins-2) * penalty_terms[i]

    # add edge weights
    for u,v in G.edges:
        i = u[0]
        G[u][v]['weight']=penalty_terms[i]

    return G

def _plot_graph(G, graph_name_file):
    num_nodes = G.number_of_nodes()
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos=pos, with_labels=True)
    plt.savefig(str(Path(__file__).parents[2] / "output" / "graph" / graph_name_file) + ".pdf")
    A = nx.to_numpy_array(G, nodelist=range(num_nodes), dtype=int)
    np.savetxt(str(Path(__file__).parents[2] / "output" / "graph" / graph_name_file) + "_adj_mat.dat", A, fmt="%d")
    
    
    