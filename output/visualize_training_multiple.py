import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


'''
Plots the data during training for different values of the depth p. The data that are

what_to_plot: the name of the type of data you want to visualize to be chosen in the
                list of what_can_be_plot
depths: which data to plot, list of integers
Nwarmup, Nbayes, trial, graph : has to be copied as they appear on the data

'''

depths = [1,2]
Nwarmup = 10
Nbayes = 100
trial = 1
graph = 14
Ntot = Nwarmup + Nbayes


what_can_be_plot = ['energy',
                    'best_energy',
                     'approx_ratio',
                     'best_approx_ratio',
                     'fidelity',
                     'best_fidelity',
                     'variance',
                     'corr_length',
                     'const_kernel',
                     'std_energies',
                     'average_distances',
                     'nit',
                     'time_opt_bayes',
                     'time_qaoa',
                     'time_opt_kernel',
                     'time_step'
                     ]
what_to_plot = ['approx_ratio', 
                'best_approx_ratio',
                'fidelity', 
                'best_fidelity',
                'corr_length',
                'const_kernel',
                'variance',
                'nit'
                ]
                
#assuming 8 plots to produce can be 
#adapted to length of what_to_plot

lines = ['.-', 'P-', 's-', '^-', 'D-']
markersize = 2
linewidth = 0.8
fig, axis = plt.subplots(2,4, figsize = (10, 6)) 
line_length = int( len(what_to_plot) / 2) 

for i, depth in enumerate(depths):

    name_ = f'lfgbs_p_{depth}_punti_{Ntot}_warmup_{Nwarmup}_train_{Nbayes}_trial_{trial}_graph_{graph}'
    folder_name = name_ + '/'
    file_name = name_+'.dat'
    df = pd.read_csv(folder_name + file_name)
    
    for j, plot_name in enumerate(what_to_plot):
    
        current_axis = axis[j // line_length, j % line_length]
        current_axis.plot(
                         df[plot_name],
                         lines[i % len(lines)],
                         color = f'C{depth}',
                         label = str(depths[i]),
                         markersize = markersize,
                         linewidth = linewidth
                         )
        current_axis.set_ylabel(plot_name)
        current_axis.set_xlabel('Steps')
        current_axis.legend()
    
        
    
            
    
plt.suptitle('Different P, Ntot = {}, warmup = {}%'.format(Ntot, Nwarmup))
plt.tight_layout()
plt.show()

    





