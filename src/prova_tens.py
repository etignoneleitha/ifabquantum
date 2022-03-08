import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
 
NUM_CHAINS = 1
 
dtype = np.float32
target = tfd.Normal(loc=dtype(0), scale=dtype(1.0))   
init_state = np.ones([NUM_CHAINS, 1], dtype=dtype)
 
samples = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=init_state,
    kernel=tfp.mcmc.SliceSampler(
        target.log_prob,
        step_size=1.0,
        max_doublings=5),
    num_burnin_steps=500,
    trace_fn=None,
    seed=1234)
  
list_of_samples = [tf.squeeze(samples).numpy().tolist()] if NUM_CHAINS == 1 else [x.numpy().tolist() for x in tf.stack(tf.squeeze(samples), axis=1)]
 
for samples in list_of_samples:
    print('Sample mean: ', np.mean(samples))
    print('Sample std: ', np.std(samples))