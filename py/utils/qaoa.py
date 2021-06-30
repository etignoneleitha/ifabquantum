import networkx as nx
from collections import Counter, defaultdict, namedtuple
from itertools import product
import pandas as pd

from tqdm import tqdm
import numpy as np

# QUANTUM
from qiskit import Aer, QuantumCircuit, execute

# VIZ
from matplotlib import pyplot as plt

from utils.default_params import *


def s2z(configuration):
    return [1 - 2 * s for s in configuration]


def z2s(configuration):
    return [(1-z)/2 for z in configuration]


def str2list(s):
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


def evaluate_cost(G,
                  configuration,
                  penalty=DEFAULT_PARAMS["penalty"],
                  basis=None):
    '''
    configuration: eigenvalues
    '''
    G_comp = nx.complement(G)
    cost = 0
    if basis == "S":
        cost = -sum(configuration)
        for edge in G_comp.edges:
            cost += penalty*(configuration[edge[0]]*configuration[edge[1]])
    elif basis == "Z":
        # cost = -(len(configuration) - sum(configuration))/2
        cost = sum(configuration)/2
        for edge in G_comp.edges:
        # cost += penalty/4*(1-configuration[edge[0]])*(1-configuration[edge[1]])
            cost += penalty/4*(configuration[edge[0]] * configuration[edge[1]]
                               - configuration[edge[0]]
                               - configuration[edge[1]])
    else:
        raise ValueError('Basis should be specified: it must be one of ["S","Z"]')
    return cost


# def classical_solution(basis=None, show_top=None):
#
#     '''
#     Runs through all 2^n possible configurations and estimates how many max cliques there are and plots one
#     '''
#
#     results = {}
#
#     if basis=="S":
#         eigenvalues = s_eigenvalues #[0, 1] i.e. eigenvalues for |0> ad |1> respectively
#     elif basis=="Z":
#         eigenvalues = s2z(s_eigenvalues) #[1,-1] i.e. eigenvalues for |0> ad |1> respectively
#     else:
#         raise ValueError('Basis should be specified: it must one of ["S","Z"]')
#
#     eigen_configurations = list(product(eigenvalues, repeat=len(G)))
#     for eigen_configuration in tqdm(eigen_configurations):
#         results[eigen_configuration] = evaluate_cost(G, eigen_configuration, basis=basis)
#
#     print('All possible solutions: \n')
#     sol = pd.DataFrame(np.unique(list(results.values()), return_counts = True)).T
#     sol.columns=["energy","occurrencies"]
#     sol["frequency"]=round(sol["occurrencies"]/sol["occurrencies"].sum()*100,0)
#     if show_top is not None:
#         print(sol.head(show_top))
#     else:
#         print(sol)
#     d = dict((k, v) for k, v in results.items() if v == np.min(list(results.values())))
#     print(f'\nThere are {len(d)} MAXCLIQUE(S) with eigenvalues configuration(s) in basis \"{basis}\": {d}')
#
#     fig = plt.subplot(1, 2, 1)
#     val, counts = np.unique(list(results.values()), return_counts = True)
#     plt.bar(val, counts)
#     plt.xlabel('Energy')
#     plt.ylabel('Counts')
#     plt.title('Statistics of solutions')
#
#     fig = plt.subplot(1, 2, 2)
#     plt.title('MaxClique')
#     colors = list(d.keys())[0]
#     pos = nx.circular_layout(G)
#     nx.draw_networkx(G, node_color=colors, node_size=200, alpha=1, pos=pos)


def quantum_algorithm(G,
                      gamma,
                      beta,
                      penalty=DEFAULT_PARAMS["penalty"]):

    N = G.number_of_nodes()
    G_comp = nx.complement(G)
    qc = QuantumCircuit(N, N)
    qc.h(range(N))

    for edge in G_comp.edges:
        qc.rzz(gamma * penalty / 2, edge[0], edge[1])
        qc.rz(-gamma * penalty / 2, edge[0])
        qc.rz(-gamma * penalty / 2, edge[1])

    for i in G.nodes:
        qc.rz(gamma, i)

    qc.rx(2 * beta, range(N))

    meas = QuantumCircuit(N,N)
    meas.barrier(range(N))
    meas.measure(range(N), range(N))

    return qc + meas


def QAOA(G,
         gamma,
         beta,
         penalty=DEFAULT_PARAMS["penalty"],
         shots=DEFAULT_PARAMS["shots"],
         basis="S"):

    '''
    Applies QAOA
    '''

    backend = Aer.get_backend("qasm_simulator")
    qc = quantum_algorithm(G, gamma, beta, penalty)
    simulate = execute(qc, backend=backend, shots=shots)
    results = simulate.result()
    extimated_f1 = 0
    counts = results.get_counts()

    pretty_counts = {k[::-1]:v for k,v in counts.items()}
    if basis=="S":
        pass
    elif basis=="Z":
        pretty_counts = {"".join([str(x) for x in s2z([int(x) for x in k])]):v for k,v in pretty_counts.items()}
    else:
        raise ValueError('Basis should be specified: it must one of ["S","Z"]')

    for configuration in pretty_counts:
        prob_of_configuration = pretty_counts[configuration]/shots
        extimated_f1 += prob_of_configuration*evaluate_cost(G,
                                                            str2list(configuration),
                                                            basis=basis)

    return extimated_f1, pretty_counts


def grid_search(G,
                num_params_grid=DEFAULT_PARAMS["num_params_grid"],
                penalty=DEFAULT_PARAMS["penalty"],
                shots=DEFAULT_PARAMS["shots"],
                show_plot=DEFAULT_PARAMS["show_plot"]):

    QAOA_results = []
    Point = namedtuple("Point", "gamma beta f1")
    lin = np.linspace(0, np.pi, num_params_grid)
    params = np.array(list((product(lin, repeat = 2))))

    X = np.unique(params[:,0])
    Y = np.unique(params[:,1])
    X, Y = np.meshgrid(X, Y)
    Q = np.zeros((len(X),len(X)))

    for i, j in tqdm(list(product(range(len(X)),repeat=2))):
        Q[i,j], _ = QAOA(G,
                         X[i,j],
                         Y[i,j],
                         penalty=penalty,
                         shots=shots,
                         basis="Z"
                         )

    if show_plot:

        plt.imshow(Q, extent = [0, np.pi, np.pi, 0])
        plt.title('Grid Search: [{} x {}]'.format(len(X), len(X)))
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)

        cb = plt.colorbar()
        plt.xlabel(r'$\gamma$', fontsize=20)
        plt.ylabel(r'$\beta$', fontsize=20)
        cb.ax.tick_params(labelsize=15)

        points = [x for x in np.dstack((X,Y,Q)).reshape(-1,3)]
        plt.show()

        return points


def plot_distribution(freq_dict):

    sorted_freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))
    color_dict = {key: 'g' for key in sorted_freq_dict}
    plt.figure(figsize=(12,6))
    plt.xlabel("configuration")
    plt.ylabel("counts")
    plt.xticks(rotation='vertical')
    plt.bar(sorted_freq_dict.keys(), sorted_freq_dict.values(), width=0.5, color = color_dict.values())


def spectrum_vs_penalty(penalty_min=-2,
                        penalty_max=3,
                        penalty_step=0.5,
                        show_plot="show_plot",
                        basis="S"):

    configuration_energies = defaultdict(list)
    penalties=np.arange(penalty_min, penalty_max, penalty_step)

    if basis == "S":
        eigenvalues = s_eigenvalues
    elif basis=="Z":
        eigenvalues = s2z(s_eigenvalues)

    for eigen_configuration in product(eigenvalues, repeat = len(G)):
        for penalty in penalties:
            configuration_energies[str(eigen_configuration)].append(evaluate_cost(str2list(eigen_configuration), penalty, basis=basis))

    degeneracies_df = pd.DataFrame(sorted(Counter([v[-1] for k,v in configuration_energies.items()]).items(), key=lambda item: item[0], reverse=False))
    degeneracies_df.columns=["energy","eigenstates"]

    if show_plot:

        for k, v in configuration_energies.items():
            plt.plot(penalties, v, label = k, marker="o")
            plt.xlabel("penalty")
            plt.ylabel("cost")
            plt.legend(title="eigen_configuration",
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left')
        plt.show()

    return degeneracies_df, configuration_energies
