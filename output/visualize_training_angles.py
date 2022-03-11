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

depth = 3
Nwarmup = 10
Nbayes = 100
trial = 1
graph = 4343
Ntot = Nwarmup + Nbayes

def angle_names_string():
    gamma_names = [f'GAMMA_{i}' for i in range(depth)]
    beta_names = [f'BETA_{i}' for i in range(depth)]
    
    angle_names = []
    for i in range(depth):
        angle_names.append(gamma_names[i])
        angle_names.append(beta_names[i])
        
    return angle_names
        
angles_names = angle_names_string()

#assuming 8 plots to produce can be 
#adapted to length of what_to_plot

lines = ['.-', 'P-', 's-', '^-', 'D-']
markersize = 2
linewidth = 0.8
fig, axis = plt.subplots(2,1, figsize = (10, 6)) 


name_ = f'lfgbs_p_{depth}_punti_{Ntot}_warmup_{Nwarmup}_train_{Nbayes}_trial_{trial}_graph_{graph}'
folder_name = name_ + '/'
file_name = name_+'.dat'
df = pd.read_csv(folder_name + file_name)
for i, angle in enumerate(angles_names):
    even_or_odd = i % 2
    p = i // 2 
    current_axis = axis[even_or_odd]
    current_axis.plot(
                     df[angle],
                     lines[even_or_odd],
                     color = f'C{p}',
                     label = str(p),
                     markersize = markersize,
                     linewidth = linewidth
                     )
    #current_axis.set_ylabel(angle)
    current_axis.set_xlabel('Steps')
    current_axis.legend()

    

axis[0].set_ylabel('Beta')
axis[1].set_ylabel('Gamma')
plt.suptitle('Different P, Ntot = {}, warmup = {}%'.format(Ntot, Nwarmup))
plt.tight_layout()
plt.show()







