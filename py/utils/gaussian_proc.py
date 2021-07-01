#BASE
#import networkx as nx
#from collections import Counter, defaultdict, namedtuple
from itertools import product
#import pandas as pd
from scipy.stats import norm
#from tqdm import tqdm
import numpy as np

#ML
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK
from sklearn.preprocessing import StandardScaler
from sklearn.utils.optimize import _check_optimize_result
from scipy.optimize import minimize

#QUANTUM
from qiskit import Aer, QuantumCircuit, execute

#VIZ
from matplotlib import pyplot as plt


from utils.default_params import *

from utils.qaoa import (QAOA)


def bayesian_opt(G,
                 penalty,
                 shots,
                 params_bayes,
                 seed = 0,
                 verbose = False):

    num_iter_max = params_bayes["num_max_iter_bayes"]
    N_train = params_bayes["N_train"]
    N_test = params_bayes["N_test"]
    acq_function = params_bayes["acq_function"]

    gamma_extremes = [0, np.pi]; beta_extremes = [0, np.pi]
    X_train = []
    y_train = []

    #seed
    if seed:
        my_seed = seed
        np.random.seed(my_seed)

    for i in range(N_train):
        X = [np.random.uniform(*gamma_extremes), np.random.uniform(*beta_extremes)]
        X_train.append(X)
        Y, _ = QAOA(G, *X)
        y_train.append(Y)

    X = np.linspace(0, np.max([gamma_extremes, beta_extremes]), N_test)
    X_test = list(product(X, X))

    X_train = rescaler(np.array(X_train)).tolist()
    X_test = rescaler(np.array(X_test)).tolist()

    #create gaussian process and fit training data
    kernel = CK(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = MyGaussianProcessRegressor(kernel=kernel,
                                    n_restarts_optimizer=9,
                                    alpha = 1e-2,
                                    normalize_y = True,
                                    max_iter = 50000)
    gp.fit(X_train, y_train)

    #At each iteration we calculate the best point where to sample from
    sample_points = []   #We save every point that was chosen to sample from

    for i in range(num_iter_max):

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

            next_point = rescaler(np.array(next_point_normalized),
                                  min_old=0,
                                  max_old=1,
                                  min_new=0,
                                  max_new=np.pi)

            X_train.append(next_point_normalized)
            y_next_point,_ = QAOA(G, *next_point)
            y_train.append(y_next_point)
            
            if verbose:
                print(i, next_point, y_next_point)

            gp.fit(X_train, y_train)
            sample_points.append(list(next_point) + [y_next_point])

    print('End process')
    return sample_points



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


def rescaler(x, min_old=0, max_old=np.pi, min_new=0, max_new=1):

    x_sc = min_new + (max_new - min_new)/(max_old - min_old)*(x - min_old)

    return x_sc