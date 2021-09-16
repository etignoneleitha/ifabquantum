import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Some plotting, all data are mod np.pi/2 for symmetry reasons to investigate
    1. For each beta acting on a different qubit we plot the histogram of its values
    2. For each set of N betas the variance is calculated and then plotted in a
    histogram
'''

#Pick the maxcut or maxclique hamiltonian
A = np.loadtxt('maxcut/Different_betas_optimal_params_N8_p04.dat')
print(A.shape)

seeds = A[:,0]
gammas = A[:,1]
betas = A[:,2:]


#Distributions of the individual betas
fig, axis = plt.subplots(nrows = 2, ncols = 4)
fig.set_size_inches(18.5, 10.5)
for i in range(len(betas[0])):
    axis[i // 4, i % 4].hist(betas[:,i] % np.pi/2)
    axis[i // 4, i % 4].set_title('beta {}, avg {:.3f}'.format(i, np.mean(betas[:,i])))
    axis[i // 4, i % 4].tick_params(axis='both', which='major', labelsize=20)
    
#Distribution of the variance of every set of betas
beta_variances = np.var(betas % np.pi/2, axis = 1)
fig = plt.figure()
values = plt.hist(beta_variances)
plt.title('N of graphs {}'.format(len(betas[:])))

plt.clf()
df = pd.DataFrame(betas, columns = ['beta {}'.format(i) for i in range(betas.shape[1])])
print(df.corr())
plt.imshow(df.corr(), cmap = 'hot', interpolation = 'nearest', vmin = -.1, vmax = .5)
plt.colorbar()
