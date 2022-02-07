import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.qaoa_pulser import *
from utils.gaussian_process import *
import time

from pathlib import Path

### TRAIN PARAMETERS
depth = 1
Nwarmup = 3
Nbayes = 50
method = 'DIFF-EVOL'
param_range = [100, 2000]   # extremes where to search for the values of gamma and beta
quantum_noise = 1

output_folder = Path(__file__).parents[1] / "output"
file_name = 'p={}_punti={}_warmup={}_train={}.dat'.format(depth, Nwarmup + Nbayes, Nwarmup, Nbayes)

data = []
global_time = time.time()
results_structure = ['iter ', 'point ', 'energy ', 'fidelity ', 'corr_length ', 'const kernel ',
                    'std energies ', 'average distances ', 'nit ', 'time opt bayes ', 'time qaoa ', 'time opt kernel ', 'time step ']

### CREATE GRAPH AND REGISTER 
pos = np.array(
				[[0., 0.],[-4, -7],[4, -7],[8, 6],[-8, 6]]
               )
               
qaoa = qaoa_pulser(pos, quantum_noise)
gs_en, gs_state, deg = qaoa.calculate_physical_gs()

### INITIAL RANDOM POINTS
X_train = []   #data
y_train = []   #label

### CREATE GP AND FIT TRAINING DATA
kernel =  ConstantKernel(1)* Matern(length_scale=0.11, length_scale_bounds=(1e-01, 100.0), nu=1.5)
gp = MyGaussianProcessRegressor(kernel=kernel,
                                n_restarts_optimizer=10,
                                param_range = param_range,
                                alpha=1e-2,
                                normalize_y=True,
                                max_iter=50000)

X_train, y_train = qaoa.generate_random_points(Nwarmup, depth, param_range)
gp.fit(X_train, y_train)


data = [[i] + x + [y_train[i], 
                    qaoa.fidelity_gs_exact(x), 
                    qaoa.fidelity_gs_sampled(x),
                    gp.kernel_.get_params()['k2__length_scale'],
                    gp.kernel_.get_params()['k1__constant_value'], 0, 0, 0, 0, 0, 0, 0
                    ] for i, x in enumerate(X_train)]
                    
#init_pos = [0.2, 0.2]*depth
print('Training ...')
print(X_train)
for i in range(Nbayes):
    start_time = time.time()
    next_point, n_it, avg_sqr_distances, std_pop_energy = gp.bayesian_opt_step(depth, method)
    bayes_time = time.time() - start_time
    y_next_point = qaoa.expected_energy(next_point)
    qaoa_time = time.time() - start_time - bayes_time
    fid_exact = qaoa.fidelity_gs_exact(next_point)
    fid_sampled = qaoa.fidelity_gs_sampled(next_point)

    corr_length = gp.kernel_.get_params()['k2__length_scale']
    constant_kernel = gp.kernel_.get_params()['k1__constant_value']
    gp.fit(next_point, y_next_point)
    kernel_time = time.time() - start_time - qaoa_time - bayes_time
    step_time = time.time() - start_time
    
    new_data = [i+Nwarmup] + next_point + [y_next_point, fid_exact, fid_sampled, corr_length, constant_kernel, 
                                    std_pop_energy, avg_sqr_distances, n_it, 
                                    bayes_time, qaoa_time, kernel_time, step_time]                    
    data.append(new_data)
    print((i+1),' / ',Nbayes)
    print(new_data)
    format = '%.d ' + (len(new_data) - 1)*'%.4f '
    np.savetxt(output_folder / file_name, data, fmt = format)
     
best_x, best_y, where = gp.get_best_point()
data.append(data[where])

np.savetxt(output_folder / file_name, np.array(data), fmt = format)
print('Best point: ' , data[where])
print('time: ',  time.time() - global_time)
