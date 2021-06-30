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

    solution_state_str = "".join(str(int(_)) for _ in solution_state)
    return G, solution_state_str


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