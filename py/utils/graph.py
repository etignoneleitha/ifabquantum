import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


# Set graph instance & its complement
def create_graph(num_nodes, edge_list):

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)

    all_cliques = list(nx.find_cliques(G))

    length_cliques = [len(clique) for clique in all_cliques]

    solution_state = np.zeros(num_nodes)

    for index_clique in all_cliques[np.argmax(length_cliques)]:
        print(index_clique)
        solution_state[index_clique] = 1

    solution_max_clique = "".join(str(int(_)) for _ in solution_state)
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
    
def create_random_graphs(total, 
                         N_nodes_min,
                         N_nodes_max, 
                         M_min,
                         max_conn):
    '''

    Parameters
    ----------
    total : int
        Number of random graphs to generate.
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
    
    graphs = []
    
    for i in range(total):
        N_nodes = np.random.choice(range(N_nodes_min, N_nodes_max +1, 1))
        M_edges = np.random.choice(range(M_min, N_nodes+1, 1))
        G = nx.gnm_random_graph(N_nodes, M_edges)
        graphs.append(G)
    return graphs
        
        