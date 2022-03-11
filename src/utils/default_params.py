
# Set global parameters
s_eigenvalues = [0, 1]
DEFAULT_PARAMS = {"penalty": 2,
                  "shots": 1000,
                  "num_grid": 20,
                  "seed" : 22,
                  "initial_length_scale" : 1,
                  "length_scale_bounds" : (0.01, 100),
                  "initial_sigma":1,
                  "constant_bounds":(0.01, 100),
                  "nu" : 1.5,
                  "max_iter_lfbgs": 50000,
                  "n_restart_kernel_optimizer":9,
                  "distance_conv_tol": 0.01
                  }
