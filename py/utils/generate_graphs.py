import numpy as np
import matplotlib.pyplot as plt

num_iterations = 20
num_features = 3
start_conditions = [1,2,5,10]
N_start_conditions = len(start_conditions)



all_data = np.zeros((4,3,100))

for i, n_train in enumerate(start_conditions):
    
    for j in range(num_iterations):
        A = np.loadtxt('../../data/raw/N_train_{}_iter_{}.dat'.format(n_train, j))
        
        for k in range(num_features):
            all_data[i, k] += A[:,k]/num_iterations
        
        
datas = ['Avg energy', 'AR', 'Prob solution']
choice = 2  #between the datas list
markers = ['+', '^', '*', 'x']

fig = plt.figure()
plt.xlabel('Step')
plt.ylabel(datas[choice])
for i in range(N_start_conditions):
    plt.plot(all_data[i, choice],
             linewidth = 1,
             markersize = 4,
             marker = markers[i],
             label = start_conditions[i])
    
plt.legend()

plt.savefig('../../data/processed/{}.pdf'.format(datas[choice]))
        