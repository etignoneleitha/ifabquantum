import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from ._differentialevolution import DifferentialEvolutionSolver
from  utils.default_params import *
# SKLEARN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.utils.optimize import _check_optimize_result
from scipy.special import ndtr
from typing import List, Tuple, Union




# Allows to change max_iter (see cell below) as well as gtol.
# It can be straightforwardly extended to other parameters
class MyGaussianProcessRegressor(GaussianProcessRegressor):
    
    def __init__(self, 
                 angles_bounds: Tuple,
                 gtol: float, 
                 max_iter: int, 
                 *args, 
                 **kwargs) -> None:
        """Initializes gaussian process class.

        Args:
            angles_bounds : range of the angles beta and gamma.
            gtol: tolerance of convergence for the optimization of the 
                  kernel parameters.
            max_iter: maximum number of iterations for the optimization of the 
                      kernel parameters.
                      
            *args, **kwargs: The parameters passed to the sklearn class for GP:
                                kernel, optimizer_kernel, n_restarts_optimizer 
                                (how many times the kernel opt is performed)
                                normalize_y: default is yes.
        """
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
        """Returns a dictionary of infos on the  gp to print
        """
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

    def print_info(self, f) -> None:
        """Prints information on the passed file f
        
        Args:
            f: file on which to write the info
        """
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
                                  obj_func: callable,
                                  initial_theta: float,
                                  bounds: Tuple) -> Tuple[float, float]:
                                  
        """Overrides the super()._constrained_optimization to perform otpimization 
        of the kernel parameters by maximizing the log marginal likelihood.
        It is only called by super().fit, so at every fitting of the training points
        or at a new bayesian opt step. Options for the optimization are fmin_l_bfgs_b,
        differential_evolution.
        
        Args:
            obj_func: the func to minimize.
            initial_theta: starting point for the optimization.
            bounds: bounds of the optimization.
            
        Returns:
            The optimal parameters and the minimum of the function.
            
        """

        def obj_func_no_grad(x):
                return  obj_func(x)[0]
        
                
        if self.optimizer == "fmin_l_bfgs_b":
            self.samples = []
            tupla = []
            tupla.append(obj_func(initial_theta)[0])
            tupla.append(obj_func(initial_theta)[1][0])
            tupla.append(obj_func(initial_theta)[1][1])
            data = np.concatenate(
                                 ([len(self.samples)], initial_theta,  tupla)
                                 )
            self.samples.append(data.tolist())
        
            def callbackF(Xi):
                tupla = []
                tupla.append(obj_func(Xi)[0])
                tupla.append(obj_func(Xi)[1][0])
                tupla.append(obj_func(Xi)[1][1])
                data = np.concatenate(
                                     ([len(self.samples)], Xi,  tupla)
                                     )
                self.samples.append(data.tolist())
                
            opt_res = minimize(obj_func,
                               initial_theta,
                               method="L-BFGS-B",
                               jac=True,
                               #callback = callbackF,
                               bounds=bounds,
                               options={'maxiter': self.max_iter,
                                        'gtol': self.gtol}
                                           )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun


        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func,
                                                 initial_theta,
                                                 bounds=bounds)
        elif self.optimizer is None:
            theta_opt = initial_theta
            func_min = obj
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min


    def fit(self, new_point: List[float], 
                  y_new_point: Union[float, List[float]]
                  ) -> None:
        """Fits the GP to the new point(s).

        Appends the new data to the myGP instance and keeps track of the best X and Y.
        Then uses the inherited fit method which optimizes the kernel (by maximizing the
        log marginal likelihood) with kernel_optimizer for 1 + n_restart_optimier_kernel
        times and keeps the best value. All points are scaled down to [0,1]*depth.

        Args:
            new_point: either list or a single new point.
            y_new_point: the energy of new_point.
            
        """
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

    def scale_down(self, point: List[float]) -> List[float]:
        """Rescales a(many) point(s) from angles bounds to [0,1].
        
        Args:
            point: point to rescale.
        """

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


    def scale_up(self, point: List[float]) -> List[float]:
        """Rescales a(many) point(s) from [0, 1] to angle bounds.
        
        Args:
            point: point to rescale.
        """

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

    def get_best_point(self) -> Tuple[List[float], float, int]:
        """Return the current best point with its energy and position.
        
        Returns:
            The best set of angles found so far by the optimization, their 
            energy and their position in the list of visited points.
        """
        
        x_best = self.scale_up(self.x_best)
        where = np.argwhere(self.y_best == np.array(self.Y))
        return x_best, self.y_best, where[0,0]

    def acq_func(self, x: List[float], *args) -> float:
        """Expected improvement at point x.

        Args:
            x: point of the prediction.
            *args: the sign of the acquisition function (in case you have 
                    to minimize -acq fun).
                    
        Returns:
            Value of the acquisition function.

        """
        try:
            sign = args[0]
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



    def bayesian_opt_step(self, 
                          method: str = 'DIFF-EVOL', 
                          init_pos: List[float] = None) -> Tuple[List[float],
                                                                 int,
                                                                 float,
                                                                 float]:
        """Performs one step of bayesian optimization using the specified 
        method to maximize the acquisition function.

        Args:
            method: differential-evolution chosen by default at the moment.

        Returns:
            next_point: where to sample next (rescaled to param range).
            nit: number of iterations of the diff evolution algorithm.
            avg_norm_dist_vect: condition of convergence for the positions.
            std_pop_energy: condition of convergence for the energies.
        """
        
        depth = int(len(self.X[0])/2)

        samples = []
        acqfunvalues = []

        if method == 'DIFF-EVOL':
            fun = self.acq_func
            diff_evol_args = [-1]
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

    def get_covariance_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the covariance matrix of the Gaussian Process
        
        Returns:
            The matrix K with its eigenvalues
        """
        K = self.kernel_(self.X)
        K[np.diag_indices_from(K)] += self.alpha
        eigenvalues, eigenvectors = np.linalg.eig(K)
        
        return K, eigenvalues
        
    def plot_covariance_matrix(self, show:bool = True, save:bool = False) -> None:
        """Plots the covariance matrix
        
        Args:
            show: if needs to show image
            save: if needs to save
        """
        K = self.get_covariance_matrix()
        fig = plt.figure()
        im = plt.imshow(K, origin = 'upper')
        plt.colorbar(im)
        if save:
            plt.savefig(f'data/cov_matrix_iter={len(self.X)}.png')
        if show:
             plt.show()

    def plot_posterior_landscape(self, show:bool = True, save:bool = False) -> None:
    
        """Plots the mean of the Gaussian Process at the current iteration.
        
        Args:
            show: if needs to show image
            save: if needs to save
        """
        
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


    def get_acquisition_function(self, show:bool = True, save:bool = False) -> np.ndarray:
        """Calculates the acquisition function at the current iteration.
        
        Args:
            show: if needs to show image
            save: if needs to save
        """
        if len(self.X[0]) > 2:
            raise ValueError(
                        "Non si puo plottare l'AF a p>1"
                    )
        fig = plt.figure()
        num = 50
        x = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                x[j, i] = self.acq_func([i/num,j/num])
        im = plt.imshow(x, extent = [0,1,0,1], origin = 'lower')

        samples = np.array(self.X)
        x_points = samples[:len(self.X), 0]
        y_points = samples[:len(self.X),1]
        plt.scatter(x_points, y_points, marker = '+', c = 'g')
        plt.scatter(samples[-1, 0], samples[-1,1], marker = '+', c = 'r')
        plt.xlabel('Gamma')
        plt.ylabel('Beta')
        for i, p in enumerate(samples):
        
            plt.annotate(f'{i}', p)
            
        plt.colorbar(im)
        plt.title('data/ACQ F iter:{} kernel_{}'.format(len(self.X), self.kernel_))
        if save:
            plt.savefig('output/acq_fun_iter={}.png'.format(len(self.X)))
        if show:
            plt.show()
        
        return x
    
    def plot_log_marginal_likelihood(self, show:bool = False, save:bool = False) -> None:
        """Plots the log marginal likelihood at the current iteration.
        
        Args:
            show: if needs to show image
            save: if needs to save
        """
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

    def get_log_marginal_likelihood_grid(self, show = False, save = False) -> np.ndarray:
        """Calculates the log marginal likelihood at the current iteration.
        
        Args:
            show: if needs to show image
            save: if needs to save
        """
        num = 50
        x = np.zeros((num, num))
        min_x = np.log(DEFAULT_PARAMS['length_scale_bounds'][0])
        max_x = np.log(DEFAULT_PARAMS['length_scale_bounds'][1])
        min_y = np.log(DEFAULT_PARAMS['constant_bounds'][0])
        max_y = np.log(DEFAULT_PARAMS['constant_bounds'][1])
        ascisse = np.linspace(min_x, max_x, num = num)
        ordinate = np.linspace(min_y, max_y, num = num)
        for i, ascissa in enumerate(ascisse):
            for j, ordinata in enumerate(ordinate):
                x[j, i] = self.log_marginal_likelihood([ascissa, ordinata])
        
        x = np.array(x)
        
        if show or save:
            im = plt.imshow(x,  extent = [min_x, max_x, min_y, max_y], origin = 'lower', aspect = 'auto')
            plt.xlabel('constant')
            plt.ylabel('corr length')
            plt.scatter(np.array(self.samples)[:,1], np.array(self.samples)[:,2])
            plt.colorbar(im)
            max = np.max(x)
            plt.clim(max-20, max*1.05)
            plt.title('log_marg_likelihood iter:{} kernel_{}'.format(len(self.X), self.kernel_))
        if save:
            plt.savefig('output/marg_likelihood_iter={}_kernel={}.png'.format(len(self.X), self.kernel_))
        if show:
            plt.show()
        
        return x
        
