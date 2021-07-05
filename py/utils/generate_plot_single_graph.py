import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph import draw_graph
import os

num_iterations = 40
features = ['Avg energy', 'AR', 'Prob solution']
choice = 1 #between the features list
num_features = len(features)

start_conditions = [1, 2, 5, 10, 20]
markers = ['+', '^', '*', 'x', 'v']

N_start_conditions = len(start_conditions)

all_data = np.zeros((N_start_conditions,
                     num_features,
                     100))

num_graph = 5

folder = "../../data/raw/"

for i, n_train in enumerate(start_conditions):
    files_n_train = []
    for filename in os.listdir(folder):
        if filename.startswith("graph_{}_N_train_{}_".format(num_graph+1, n_train)):
            files_n_train.append(filename)

    num_iterations = len(files_n_train)
    print(n_train, num_iterations)
    for file_n_train in files_n_train:
        A = np.loadtxt(folder + file_n_train)
        for k in range(num_features):
            all_data[i, k] += A[:,k] / num_iterations

# exit()
# if 0:
#     for j in range(num_iterations):
#         A = np.loadtxt("../../data/raw/graph_{}_N_train_{}_iter_{}_N_average_40.dat".format(num_graph+1, n_train, j))
#         print("../../data/raw/graph_{}_N_train_{}_iter_{}_N_average_40.dat".format(num_graph+1, n_train, j))
#         for k in range(num_features):
#             all_data[i, k] += A[:,k]/num_iterations
#
# print(all_data.shape)

fig = plt.figure()
plt.xlabel('Step')
plt.ylabel(features[choice])
average = np.zeros(100)

# Single graph

for i in range(N_start_conditions):
    std_error = np.std(all_data[i, choice]) / np.sqrt(num_iterations)
    x = range(len(all_data[i, choice]))
    plt.errorbar(x,
                 all_data[i, choice],
                 yerr=std_error,
                 linewidth = 1,
                 markersize = 4,
                 marker = markers[i],
                 label = start_conditions[i])

plt.legend()
folder = "../../data/processed/"
filenameplot = f"graph_number_{num_graph}_result_{features[choice]}.pdf"
plt.savefig(folder + filenameplot)

plt.cla()
file_pickle = nx.read_gpickle(f"../../data/processed/networkx_graphs/{num_graph+1}.gpickle")
draw_graph(file_pickle)
folder = "../../data/processed/"
filenameplot = f"graph_number_{num_graph}.pdf"
plt.savefig(folder + filenameplot)
