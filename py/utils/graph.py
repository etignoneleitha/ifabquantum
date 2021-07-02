import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import os

PATH_GRAPHS = "../data/processed/networkx_graphs/"


def find_maxclique(G):
    all_cliques = list(nx.find_cliques(G))
    num_nodes = G.number_of_nodes()
    length_cliques = [len(clique) for clique in all_cliques]

    solution_state = np.zeros(num_nodes)

    for index_clique in all_cliques[np.argmax(length_cliques)]:
        print(index_clique)
        solution_state[index_clique] = 1

    solution_max_clique = "".join(str(int(_)) for _ in solution_state)
    
    return solution_max_clique


def create_graph(num_nodes, edge_list):

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)

    solution_max_clique = find_maxclique(G)
    return G, solution_max_clique


def load_graph_pickle(key_graph):
    filename = PATH_GRAPHS + "{}.gpickle".format(key_graph) 
    G = nx.read_gpickle(filename)

    solution_max_clique = find_maxclique(G)
    return G, solution_max_clique


def draw_graph(G):

    G_comp = nx.complement(G)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes[0].set_title("GRAPH")
    axes[1].set_title("COMPLEMETARY GRAPH")
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, ax=axes[0], pos=pos)
    pos = nx.circular_layout(G_comp)
    nx.draw_networkx(G_comp,
                     ax=axes[1],
                     node_color="r",
                     pos=pos)
    
def create_random_graphs(N_nodes_min,
                         N_nodes_max, 
                         M_min,
                         max_conn,
                         key_graph):
    '''

    Parameters
    ----------
    N_min : int
        minimum number of nodes
    N_max : int
        maximum number of nodes
    M_min : int
        minimum number of edges
    max_conn : int
        max connectivity

    Returns
    -------
    Tuple of graphs N, edges

    '''
    c = 0
    while c == 0:
        N_nodes = np.random.choice(range(N_nodes_min, N_nodes_max +1, 1))
        M_edges = np.random.choice(range(M_min, N_nodes+1, 1))
        G = nx.gnm_random_graph(N_nodes, M_edges)
        
        degree_list = np.array(G.degree())
        if max(degree_list[:,1]) > max_conn:   #checks connectivity
            c=0
        else:
            c = 1
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = PATH_GRAPHS + "{}.gpickle".format(key_graph)
    nx.write_gpickle(G, filename)
    
    return G
        
        
