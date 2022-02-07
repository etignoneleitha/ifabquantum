import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from ._differentialevolution import DifferentialEvolutionSolver
from .HMC import *
from .grad_descent import *
from  utils.default_params import *
# SKLEARN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from itertools import product
from sklearn.utils.optimize import _check_optimize_result
from scipy.stats import norm
from scipy.special import ndtr
from sklearn.preprocessing import StandardScaler
import random
import dill
import warnings
#import zeus
# warnings.filterwarnings("error")
from sklearn.exceptions import ConvergenceWarning
from pathlib import Path

import tensorflow as tf

import tensorflow_probability as tfp


# Allows to change max_iter (see cell below) as well as gtol.
# It can be straightforwardly extended to other parameters
class MyGaussianProcessRegressor(GaussianProcessRegressor):
    
    def __init__(self, angles_bounds, gtol, max_iter, *args, **kwargs):
        '''Initializes gaussian process class

        The class also inherits from Sklearn GaussianProcessRegressor
        Attributes for MYgp
        --------
        angles_bounds : range of the angles beta and gamma

        gtol: tolerance of convergence for the optimization of the kernel parameters

        max_iter: maximum number of iterations for the optimization of the kernel params

        Attributes for SKlearn GP:
        ---------

        *args, **kwargs: kernel, optimizer_kernel,
                         n_restarts_optimizer (how many times the kernel opt is performed)
                         normalize_y: standard is yes
        '''
        alpha = 10e-10
        super().__init__(alpha = alpha, *args, **kwargs)
        self.max_iter = max_iter
        self.gtol = gtol
        self.angles_bounds = angles_bounds
        self.X = []
        self.Y = []
        self.x_best = 0
        self.y_best = np.inf
        self.seed = DEFAULT_PARAMS["seed"]
        self.samples = []

    def get_info(self):
        '''
        Returns a dictionary of infos on the  gp to print
        '''
        info ={}
        info['param_range'] = self.angles_bounds
        info['acq_fun_optimization_max_iter'] = self.max_iter
        info['seed'] = self.seed
        info['gtol'] = self.gtol
        info['alpha'] = self.alpha
        info['kernel_optimizer'] = self.optimizer
        info['kernel_info'] = self.kernel.get_params()
        info['n_restart_kernel_optimizer'] = self.n_restarts_optimizer
        info['normalize_y'] = self.normalize_y

        return info

    def print_info(self, f):
        f.write(f'parameters range: {self.angles_bounds}\n')
        f.write(f'acq_fun_optimization_max_iter: {self.optimizer}\n')
        f.write(f'seed: {self.seed}\n')
        f.write(f'tol opt kernel: {self.gtol}\n')
        f.write(f'energy noise alpha 1/sqrt(N): {self.alpha}\n')
        f.write(f'kernel_optimizer: {self.optimizer}\n')
        f.write(f'kernel info: {self.kernel.get_params()}\n')
        f.write(f'n_restart_kernel_optimizer: {self.n_restarts_optimizer}\n')
        f.write(f'normalize_y: {self.normalize_y}\n')
        f.write('\n')

    def _constrained_optimization(self,
                                  obj_func,
                                  initial_theta,
                                  bounds):
        '''
        Overrides the super()._constrained_optimization to perform otpimization of the kernel
        parameters by maximizing the log marginal likelihood.
        It is only called by super().fit, so at every fitting of the training points
        or at a new bayesian opt step. Options for the optimization are fmin_l_bfgs_b,
        differential_evolution or monte_carlo. The latter just averages over M = 30 values
        of the hyperparameters and returns this average.

        Do not change the elif option
        '''

        def obj_func_no_grad(x):
                return  obj_func(x)[0]
        
                
        if self.optimizer == "fmin_l_bfgs_b":
            self.samples = []
        
            def callbackF(Xi):
                tupla = []
                tupla.append(obj_func(Xi)[0])
                tupla.append(obj_func(Xi)[1][0])
                tupla.append(obj_func(Xi)[1][1])
                data = np.concatenate(
                                     ([len(self.samples)], Xi,  tupla)
                                     )
                self.samples.append(data)
                
            opt_res = minimize(obj_func,
                               initial_theta,
                               method="L-BFGS-B",
                               jac=True,
                               callback = callbackF,
                               bounds=bounds,
                               options={'maxiter': self.max_iter,
                                        'gtol': self.gtol}
                                           )
            _check_optimize_result("lbfgs", opt_res)
            self.samples = np.array(self.samples)
            theta_opt, func_min = opt_res.x, opt_res.fun


        elif self.optimizer == 'differential_evolution':
            diff_evol = DifferentialEvolutionSolver(obj_func_no_grad,
                                            x0 = initial_theta,
                                            bounds = bounds,
                                            popsize = 15,
                                            tol = .001,
                                            dist_tol = 0.01,
                                            seed = self.seed)# as diff_evol:
            results,average_norm_distance_vectors, std_population_energy, conv_flag = diff_evol.solve()
            theta_opt, func_min = results.x, results.fun

        elif self.optimizer == 'monte_carlo':
            M = 30
            hyper_params = self.pick_hyperparameters(M, DEFAULT_PARAMS['length_scale_bounds'], DEFAULT_PARAMS['constant_bounds'])
            theta_opt = np.mean(np.log(hyper_params), axis = 0)
            func_min = obj_func_no_grad(theta_opt)

        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func,
                                                 initial_theta,
                                                 bounds=bounds)
        elif self.optimizer is None:
            theta_opt = initial_theta
            func_min = 0
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        print('L ottimiz ritorna il valore migliore ', theta_opt)
        return theta_opt, func_min

    def fit(self, new_point, y_new_point):
        '''Fits the GP to the new point(s)

        Appends the new data to the myGP instance and keeps track of the best X and Y.
        Then uses the inherited fit method which optimizes the kernel (by maximizing the
        log marginal likelihood) with kernel_optimizer for 1 + n_restart_optimier_kernel
        times and keeps the best value. All points are scaled down to [0,1]*depth.

        Attributes
        ---------
        new_point, y_new_point: either list or a single new point with their/its energy
        '''
        new_point = self.scale_down(new_point)

        if isinstance(new_point[0], float): #check if its only one point
            self.X.append(new_point)
            self.Y.append(y_new_point)
            if y_new_point < self.y_best:
                self.y_best = y_new_point
                self.x_best = new_point
        else:
            for i, point in enumerate(new_point):
                self.X.append(point)
                self.Y.append(y_new_point[i])

                if y_new_point[i] < self.y_best:
                    self.y_best = y_new_point[i]
                    self.x_best = point

        super().fit(self.X, self.Y)

    def scale_down(self, point):
        'Rescales a(many) point(s) from angles bounds to [0,1]'

        min_gamma, max_gamma=self.angles_bounds[0]
        min_beta,  max_beta = self.angles_bounds[1]

        norm = []
        if isinstance(point[0], float) or isinstance(point[0], int):
            for a,i in enumerate(point):
                if a%2 == 0:
                    norm.append(1/(max_gamma - min_gamma)*(i - min_gamma))
                else:
                    norm.append(1/(max_beta - min_beta)*(i - min_beta))

        else:
            for x in point:
                b = []
                for a,i in enumerate(x):
                    if a%2 == 0:
                        b.append(1/(max_gamma - min_gamma)*(i - min_gamma))
                    else:
                        b.append(1/(max_beta - min_beta)*(i - min_beta))
                norm.append(b)

        return norm


    def scale_up(self, point):
        'Rescales a(many)point(s) from [0,1] to angles bounds'

        min_gamma, max_gamma=self.angles_bounds[0]
        min_beta,  max_beta = self.angles_bounds[1]

        norm = []
        if isinstance(point[0], float) or isinstance(point[0], int):
            for a,i in enumerate(point):
                if a%2 == 0:
                    norm.append(min_gamma + i*(max_gamma - min_gamma))
                else:
                    norm.append(min_beta + i*(max_beta - min_beta))
        else:
            for x in point:
                b = []
                for a,i in enumerate(x):
                    if a%2 == 0:
                        b.append(round(min_gamma + i*(max_gamma - min_gamma)))
                    else:
                        b.append(round(min_beta + i*(max_beta - min_beta)))
                norm.append(b)

        return norm

    def get_best_point(self):
        '''Return the current best point with its energy and position'''
        x_best = self.scale_up(self.x_best)
        where = np.argwhere(self.y_best == np.array(self.Y))
        return x_best, self.y_best, where[0,0]

    def acq_func(self, x, *args):
        '''Expected improvement at point x

        Arguments
        ---------
        x: point of the prediction
        *args: the sign of the acquisition function (in case you have to minimize -acq fun)

        '''
        self = args[0]
        try:
            sign = args[1]
        except:
            sign = 1.0
        if isinstance(x[0], float):
            x = np.reshape(x, (1, -1))
        f_x, sigma_x = self.predict(x, return_std=True)

        f_prime = self.y_best #current best value

        #Ndtr is a particular routing in scipy that computes the CDF in half the time
        cdf = ndtr((f_prime - f_x)/sigma_x)
        pdf = 1/(sigma_x*np.sqrt(2*np.pi)) * np.exp(-((f_prime -f_x)**2)/(2*sigma_x**2))
        alpha_function = (f_prime - f_x) * cdf + sigma_x * pdf
        return sign*alpha_function


    def pick_hyperparameters(self, N_points, bounds_elle, bounds_sigma):
        ''' Generates array of (N_points,2) random hyperaparameters inside given bounds
        '''
        '''
        elle = np.random.uniform(low = bounds_elle[0], high=bounds_elle[1], size=(N_points,))
        sigma = np.random.uniform(low = bounds_sigma[0], high=bounds_sigma[1], size=(N_points,))
        hyper_params = list(zip(elle,sigma))

        lml = np.array([self.log_marginal_likelihood(np.log(hyper_param)) for hyper_param in hyper_params])
        idx_max_values = np.argpartition(lml, -50)[-50:] #takes the 50 indexes with largest value of lml
        best_params = hyper_params[idx_max_values]
        '''

        '''
            ZEUS
        nsteps, nwalkers, ndim = 100, 10, len(self.kernel_.theta)
        start = np.random.uniform(0.1,1,(nwalkers,ndim))

        sampler = zeus.EnsembleSampler(nwalkers, ndim, self.log_marginal_likelihood)
        print("start sampli")
        sampler.run_mcmc(start, nsteps)
        print("end sampli")
        hyper_params = sampler.get_chain(flat=True)[:N_points]
        print("hhhh")
        print(hyper_params)
#         try:
#             positive_hyper_params = hyper_params[hyper_params[:,0] > 0][:N_points]
#         except:
#              positive_hyper_params = hyper_params[hyper_params[:,0] > 0]
#              print('Only {} hyper_params where selected'.format(len(positive_hyper_params)))
        return hyper_params
        '''
        dtype = np.float32
        print('begin slice sampling')
        samples = tfp.mcmc.sample_chain(
                                        num_results=100,
                                        current_state=np.ones(len(self.kernel_.theta)),
                                        kernel=tfp.mcmc.SliceSampler(
                                            self.log_marginal_likelihood,
                                            step_size=1.0,
                                            max_doublings=5),
                                        num_burnin_steps=50,
                                        trace_fn=None), 
        # tfd = tfp.distributions
# 
#         kernel = tfp.mcmc.SliceSampler(tfd.Distribution(self.log_marginal_likelihood), step_size=1.0,  max_doublings=5)
#         state = tf.constant([1,1], dtype = dtype)
#         extra = kernel.bootstrap_results(state)
#         samples = []
#         for _ in range(10):
#           state, extra = kernel.one_step(state, extra)
#           samples.append(state)
        print('end slice sampling')
        samples = samples.numpy()
        samples = samples[-N_points:] #taking the last N_points
        
        return samples
        

    def mc_acq_func(self, x, *args):
        ''' Averages the value of the acq_func for different sets of hyperparameters chosen by pick_hyperparameters
        method '''

        hyper_params = args[-1]
        acq_func_values = []
        for params in hyper_params:
            self.kernel.set_params(**{'k1__constant_value': params[0]})
            self.kernel.set_params(**{'k2__length_scale': params[1]})
            acq_func_values.append(self.acq_func(x, *args))

        return np.mean(acq_func_values)

    def bayesian_opt_step(self, method = 'DIFF-EVOL', init_pos = None):
        ''' Performs one step of bayesian optimization

        It uses the specified method to maximize the acquisition function

        Attributes
        ----------
        method: choice between grid search (not available for depth >1), hamiltonian monte
                carlo, finite differences gradient, scipy general optimization, diff evolution

        Returns (for diff evolution)
        -------
        next_point: where to sample next (rescaled to param range)
        nit: number of iterations of the diff evolution algorithm
        avg_norm_dist_vect: condition of convergence for the positions
        std_pop_energy: condition of convergence for the energies
        '''
        depth = int(len(self.X[0])/2)

        samples = []
        acqfunvalues = []

        def callbackF(Xi, convergence):
            samples.append(Xi.tolist())
            acqfunvalues.append(self.acq_func(Xi, self, 1)[0])

        if method == 'GRID_SEARCH':
            if depth >1:
                print('PLS No grid search with p > 1')
                raise Error
            X= np.linspace(0, 1, 50, dtype = int)
            Y= np.linspace(0,1, 50, dtype = int)
            X_test = list(product(X, Y))


            alpha = self.acq_func(X_test, self, 1)
            #argmax is a number between 0 and N_test**-1 telling us where is the next point to sample
            argmax = np.argmax(np.round(alpha, 3))
            next_point = X_test[argmax]

        if method == 'HMC':
            path_length = .2
            step_size = .02
            epsilon = .001
            epochs = 10
            hmc_samples, traj, success= HMC(func = self.acq_func,
                                            path_len=path_length,
                                            step_size=step_size,
                                            epsilon = epsilon,
                                            epochs=epochs)
            cols = int(len(init_pos)*4)
            rows = int((path_length/step_size+1)*(epochs))
            traj_to_print = np.reshape(traj, (rows, cols))
            np.savetxt('Trajectories.dat',
                        traj_to_print,
                        header='[q_i, q_f],  [p_i, p_f],  self.acq_func(q1), dVdQ_1, dVdq_2',
                        fmt='  %1.3f')

            accepted_samples = hmc_samples[success == True]
            if len(accepted_samples) == 0:
                print('Warning: there where 0 accepted samples. Restarting with new values')
                next_point = 0

            if accepted_samples.shape[0] > 0:
                next_point = np.mean(accepted_samples[-10:], axis = 0)
            else:
                next_point = [0,0]

            if epochs< 100:
                plot_trajectories(self.acq_func, traj, success, save = True)

        if method == 'FD':
            l_rate = 0.001
            if init_pos is None:
                print('You need to pass an initial point if using Finite differences')
                raise Error
            gd_params = gradient_descent(self.acq_func,
                                        l_rate,
                                        init_pos,
                                        max_iter = 400,
                                        method = method,
                                        verbose = 1)
            next_point = gd_params[-1][:2]

        if method == 'SCIPY':
            if init_pos is None:
                print('You need to pass an initial point if using scipy')
                raise Error
            results = minimize(self.acq_func,
                                bounds = [(0,1), (0,1)]*depth,
                                x0 = init_pos,
                                args = (self, -1))
            next_point = results.x

        if method == 'DIFF-EVOL':
            if DEFAULT_PARAMS['diff_evol_func'] == 'mc':
                best_params = np.array(self.pick_hyperparameters(50, DEFAULT_PARAMS['length_scale_bounds'], DEFAULT_PARAMS['constant_bounds']))
                diff_evol_args = (self, -1, best_params)
                fun = self.mc_acq_func
            else:
                fun = self.acq_func
                diff_evol_args = (self, -1)
            with DifferentialEvolutionSolver(fun,
                                            bounds = [(0,1), (0,1)]*depth,
                                            callback = None,
                                            maxiter = 100*depth,
                                            popsize = 15,
                                            tol = .001,
                                            dist_tol = DEFAULT_PARAMS['distance_conv_tol'],
                                            seed = self.seed,
                                            args = diff_evol_args) as diff_evol:
                results,average_norm_distance_vectors, std_population_energy, conv_flag = diff_evol.solve()
            next_point = results.x
        next_point = self.scale_up(next_point)
        return next_point, results.nit, average_norm_distance_vectors, std_population_energy

    def covariance_matrix(self):
        K = self.kernel_(self.X)
        K[np.diag_indices_from(K)] += self.alpha
        eigenvalues, eigenvectors = np.linalg.eig(K)
        print(K)
        print(eigenvalues)
        
        return K, eigenvalues

    def plot_covariance_matrix(self, show = True, save = False):
        K = self.covariance_matrix()
        fig = plt.figure()
        im = plt.imshow(K, origin = 'upper')
        plt.colorbar(im)
        if save:
            plt.savefig('data/cov_matrix_iter={}.png'.format(len(self.X), self.kernel_))
        if show:
             plt.show()

    def plot_posterior_landscape(self, show = True, save = False):
        if len(self.X[0]) > 2:
            raise ValueError(
                        "Non si puo plottare il landscape a p>1"
                    )

        fig = plt.figure()
        num = 100
        x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                x[j, i] = self.predict(np.reshape([i/num,j/num], (1, -1)))
                #NOTARE LO scambio di j, i necessario per fare in modo che gamma sia x e beta sia y!
        im = plt.imshow(x, extent = [0,1,0,1], origin = 'lower')
        samples = np.array(self.X)
        im2 = plt.scatter(samples[:, 0], samples[:,1], marker = '+', c = self.Y, cmap = 'Reds')
        plt.title('Landscape at {} sampled points'.format(len(self.X)))
        plt.xlabel('Gamma')
        plt.ylabel('Beta')
        plt.colorbar(im)
        plt.colorbar(im2)
        plt.show()


    def plot_acquisition_function(self, show = True, save = False):
        if len(self.X[0]) > 2:
            raise ValueError(
                        "Non si puo plottare l'AF a p>1"
                    )
        fig = plt.figure()
        num = 50
        x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                x[j, i] = self.acq_func([i/num,j/num],self, 1)
        im = plt.imshow(x, extent = [0,1,0,1], origin = 'lower')

        samples = np.array(self.X)
        plt.scatter(samples[:len(self.X), 0], samples[:len(self.X),1], marker = '+', c = 'g')
        plt.scatter(samples[-1, 0], samples[-1,1], marker = '+', c = 'r')
        plt.colorbar(im)
        plt.title('data/ACQ F iter:{} kernel_{}'.format(len(self.X), self.kernel_))

        if save:
            plt.savefig('data/acq_fun_iter={}.png'.format(len(self.X), self.kernel_))
        if show:
            plt.show()
    
    def plot_log_marginal_likelihood(self, show = True, save = False):
        fig = plt.figure()
        num = 50
        x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                x[j, i] = self.log_marginal_likelihood([np.log((i+0.001)*2/num),np.log((j+0.001)*10/num)])
        min_x = DEFAULT_PARAMS['length_scale_bounds'][0]
        max_x = DEFAULT_PARAMS['length_scale_bounds'][1]
        min_y = DEFAULT_PARAMS['constant_bounds'][0]
        max_y = DEFAULT_PARAMS['constant_bounds'][1]
        im = plt.imshow(x, extent = [min_x, max_x, min_y, max_y], origin = 'lower', aspect = 'auto')
        plt.xlabel('Corr length')
        plt.ylabel('Constant')
        plt.colorbar(im)
        max = np.max(x)
        plt.clim(max-5, max*1.1)
        plt.title('log_marg_likelihood iter:{} kernel_{}'.format(len(self.X), self.kernel_))
        if save:
            plt.savefig('data/marg_likelihood_iter={}_kernel={}.png'.format(len(self.X), self.kernel_))
        if show:
            plt.show()

    def get_log_marginal_likelihood_grid(self, show = True, save = False):
        num = 50
        x = np.zeros((num, num))
        min_x = DEFAULT_PARAMS['length_scale_bounds'][0]
        max_x = DEFAULT_PARAMS['length_scale_bounds'][1]
        min_y = DEFAULT_PARAMS['constant_bounds'][0]
        max_y = DEFAULT_PARAMS['constant_bounds'][1]
        ascisse = np.linspace(np.log(min_x), np.log(max_x), num = num)
        ordinate = np.linspace(np.log(min_y), np.log(max_y), num = num)
        for i, ascissa in enumerate(ascisse):
            for j, ordinata in enumerate(ordinate):
                x[j, i] = self.log_marginal_likelihood([ascissa, ordinata])
        
        x = np.array(x)
        return x
        
