import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph import draw_graph

num_iterations = 20
num_features = 3
num_graphs = 10
start_conditions = [1,2,5,10]
N_start_conditions = len(start_conditions)
'''
for g in range(num_graphs):
    file_pickle = nx.read_gpickle(f"../../data/processed/networkx_graphs/{g+1}.gpickle")
    print(file_pickle)
    draw_graph(file_pickle)
'''
all_data = np.zeros((10, 4,3,100))




for g in range(num_graphs):
    for i, n_train in enumerate(start_conditions):
        
        for j in range(num_iterations):
            A = np.loadtxt("../../data/raw/graph_{}_N_train_{}_iter_{}.dat".format(g+1, n_train, j))
            
            for k in range(num_features):
                all_data[g, i, k] += A[:,k]/num_iterations
        
print(all_data.shape)

datas = ['Avg energy', 'AR', 'Prob solution']
choice = 1 #between the datas list
markers = ['+', '^', '*', 'x']

fig = plt.figure()
plt.xlabel('Step')
plt.ylabel(datas[choice])
average = np.zeros(100)

#Media sui grafi
for graph_number in range(num_graphs):
    i  = 3
    average += all_data[graph_number, i, choice]/num_graphs
    
#Single graph
graph_choice = 5
for i in range(N_start_conditions):
    plt.plot(all_data[graph_choice, i, choice],
             linewidth = 1,
             markersize = 4,
             marker = markers[i],
             label = start_conditions[i])
    
    
#plt.plot(average)
plt.legend()

#plt.savefig('../../data/processed/{}.pdf'.format(datas[choice]))