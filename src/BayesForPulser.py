#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import igraph
import networkx as nx
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2

from scipy.optimize import minimize


# In[6]:


def pos_to_graph(pos, d = Chadoq2.rydberg_blockade_radius(1)): #d is the rbr
    g=igraph.Graph()
    edges=[]
    for n in range(len(pos)-1):
        for m in range(n+1, len(pos)):
            pwd = ((pos[m][0]-pos[n][0])**2+(pos[m][1]-pos[n][1])**2)**0.5
            if pwd < d:
                edges.append([n,m])                         # Below rbr, vertices are connected
    g.add_vertices(len(pos))
    g.add_edges(edges)
    return g


# In[7]:


def quantum_loop(param, r):
    seq = Sequence(r, Chadoq2)
    seq.declare_channel('ch0','rydberg_global')
    middle = int(len(param)/2)
    param = np.array(param)*1 #wrapper
    t = param[:middle] #associated to H_c
    tau = param[middle:] #associated to H_0
    p = len(t)
    for i in range(p):
        ttau = int(tau[i]) - int(tau[i]) % 4
        tt = int(t[i]) - int(t[i]) % 4
        pulse_1 = Pulse.ConstantPulse(ttau, 1., 0, 0) # H_M
        pulse_2 = Pulse.ConstantPulse(tt, 1., 1, 0) # H_M + H_c
        seq.add(pulse_1, 'ch0')
        seq.add(pulse_2, 'ch0')
    seq.measure('ground-rydberg')
    simul = Simulation(seq, sampling_rate=.1)
    results = simul.run()
    count_dict = results.sample_final_state(N_samples=1000) #sample from the state vector
    return count_dict


# In[8]:


def plot_distribution(C):
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    color_dict = {key: 'g' for key in C}
    indexes = ['01011', '00111']  # MIS indexes
    for i in indexes:
        color_dict[i] = 'red'
    plt.figure(figsize=(12,6))
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color = color_dict.values())
    plt.xticks(rotation='vertical')
    plt.show()


# In[9]:


def get_cost_colouring(z,G,penalty=10):
    """G: the graph (igraph)
       z: a binary colouring
       returns the cost of the colouring z, depending on the adjacency of the graph"""
    cost = 0
    A = G.get_adjacency()
    z = np.array(tuple(z),dtype=int)
    for i in range(len(z)):
        for j in range(i,len(z)):
            cost += A[i][j]*z[i]*z[j]*penalty # if there's an edge between i,j and they are both in |1> state.

    cost -= np.sum(z) #to count for the 0s instead of the 1s
    return cost


# In[10]:


def get_cost(counter,G):
    cost = 0
    for key in counter.keys():
        cost_col = get_cost_colouring(key,G)
        cost += cost_col * counter[key]
    return cost / sum(counter.values())


# In[11]:


def func(param):
    #G = args[0]
    C = quantum_loop(param, r=reg)
    cost = get_cost(C,G)
    return cost


# In[12]:


pos = np.array([[0., 0.], [-4, -7], [4,-7], [8,6], [-8,6]])   # Grafo di pasqal originale
#pos = np.array([[0,0], [0,10],[0,-10],[10,0],[-10,0]])
G = pos_to_graph(pos)

qubits = dict(enumerate(pos))
reg = Register(qubits)
#reg.draw()


# ### Bayesian Opt

# In[9]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from itertools import product

from sklearn.utils.optimize import _check_optimize_result
from scipy.optimize import minimize

#Allows to change max_iter (see cell below) as well as gtol. It can be straightforwardly extended to other parameters
class MyGaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=2e05, gtol=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter, 'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min

def my_rescaler(x, min_old=1000, max_old=10000, min_new=0, max_new=1):

    x_sc = min_new + (max_new - min_new)/(max_old - min_old)*(x - min_old)

    return x_sc


# In[10]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from itertools import product
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

#decide acquisition function
acq_function = 'EI'
N_train = 10
N_test = 50 #Number of test elements
iterations = 80
gamma_extremes = [1000,10000]  #extremes where to search for the values of gamma and beta
beta_extremes = [1000,10000]

#create dataset: We start with N random points
X_train = []   #data
y_train = []   #label

for i in range(N_train):
    X = [np.random.randint(gamma_extremes[0],gamma_extremes[1]), np.random.randint(beta_extremes[0],beta_extremes[1])]
    X_train.append(X)
    Y = func(X)
    y_train.append(Y)

X= np.linspace(gamma_extremes[0], gamma_extremes[1], N_test, dtype = int)
Y= np.linspace(beta_extremes[0], beta_extremes[1], N_test, dtype = int)
X_test = list(product(X, Y))

X_train = list(my_rescaler(np.array(X_train)))
X_test = list(my_rescaler(np.array(X_test)))

#create gaussian process and fit training data
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = MyGaussianProcessRegressor(kernel=kernel,
                                n_restarts_optimizer=9,
                                alpha=1e-2,
                                normalize_y = True,
                                max_iter = 50000)
gp.fit(X_train, y_train)

if 0:
    next_point = [8714, 8714]
    count_dict = quantum_loop(next_point, r = reg)
    plot_distribution(count_dict)

    exit()
#At each iteration we calculate the best point where to sample from
sample_points = []   #We save every point that was chosen to sample from
for i in range(iterations):
        # Test GP
        new_mean, new_sigma = gp.predict(X_test, return_std=True)

        #New_mean and new_sigma both are (N_test**2, ) arrays not reshaped yet
        mean_max = np.max(new_mean)
        mean_min = np.min(new_mean)

        #Now calculate acquisitition fn as the cumulative for every point centered around the maximum
        cdf = norm.cdf(x = new_mean, loc =  mean_max, scale = new_sigma)
        pdf = norm.pdf(x = new_mean, loc =  mean_min, scale = new_sigma)

        #The qdf is instead the probability of being lower then the lowest value of the mean (where we wanto to pick the next_value)
        qdf = 1-norm.cdf(x = new_mean, loc =  mean_min, scale = new_sigma)

        if acq_function == 'PI':
            #Next values is calculated as so just because argmax returns a number betwenn 1 and n_test instead of inside the interval
            value = np.argmax(qdf)
            next_point = X_test[value]

        if acq_function == 'EI':
            alpha_function = (new_mean - mean_min - 0.001)*qdf + new_sigma*pdf
            #argmax is a number between 0 and N_test**-1 telling us where is the next point to sample
            argmax = np.argmax(np.round(alpha_function, 3))
            next_point_normalized = X_test[argmax]

        next_point = my_rescaler(np.array(next_point_normalized), min_old=0, max_old=1, min_new=1000, max_new=10000).astype(int)
        print(i, next_point, next_point_normalized)
        X_train.append(next_point_normalized)
        y_next_point = func(next_point)
        y_train.append(y_next_point)
        gp.fit(X_train, y_train)
        sample_points.append(next_point)


# In[14]:

next_point = [3755, 4673]
count_dict = quantum_loop(next_point, r = reg)
plot_distribution(count_dict)


# In[3]:

# In[ ]:




