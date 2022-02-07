import numpy as np
import random
import scipy.stats as st
import scipy.integrate as integ
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision = 5)

def normal(x,mu,sigma):
    numerator = np.exp(-1*((x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def neg_log_prob(x,mu,sigma):
    return -1*np.log(normal(x=x,mu=mu,sigma=sigma))

def leapfrog_step_old(q1, p1, step_size, epsilon, func):
    dVdQ_1 = (-1*np.log(func(q1 + epsilon)) + np.log(func(q1)))/epsilon   #First gradient (ref https://www.jarad.me/courses/stat615/slides/Hierarchical/Hierarchical5.pdf)
    p1 -= step_size*dVdQ_1/2 # as potential energy increases, kinetic energy decreases, half-step
    q1 += step_size*p1 # position increases as function of momentum 
    
    dVdQ_2 = (-1*np.log(func(q1 + epsilon)) + np.log(func(q1)))/epsilon 
    p1 -= step_size*dVdQ_2/2 # second half-step "leapfrog" update to momentum 
    return q1, p1
    

def leapfrog_step(q1, p1, step_size, epsilon, func):
    
    grad_func = []
    for i in range(len(q1)):
        epsilon_vec = np.zeros(len(q1))
        epsilon_vec[i] = epsilon
               
        A_2 = func(q1 + epsilon_vec)
        A_1 = func(q1)

        grad_func.append(-(A_2 - A_1) / (epsilon * A_1))
    dVdQ_1 = np.array(grad_func).reshape(len(q1))

    p1 -= step_size * dVdQ_1/2 
    
    q1 += step_size * p1 

    grad_func = []
    for i in range(len(q1)):
        epsilon_vec = np.zeros(len(q1))
        epsilon_vec[i] = epsilon
               
        A_2 = func(q1 + epsilon_vec)
        A_1 = func(q1)

        grad_func.append(-(A_2 - A_1) / (epsilon * A_1))
        
    dVdQ_2 = np.array(grad_func).reshape(len(q1))
    p1 -= step_size * dVdQ_2/2 # second half-step "leapfrog" update to momentum 
    return q1, p1, A_1, step_size *dVdQ_1/2 , step_size*dVdQ_2/2
    
    
def metropolis_acceptance(func, q0, p0, q1, p1):
    q0_nlp = func(q0)
    q1_nlp = func(q1)

    p0_nlp = normal(x=p0,mu=0,sigma=1)
    p1_nlp = normal(x=p1,mu=0,sigma=1)

    target = q1_nlp/q0_nlp # P(q1)/P(q0)
    adjustment = p1_nlp/p0_nlp # P(p1)/P(p0)
    acceptance = target*adjustment
    
    event = random.uniform(0,1)
    return np.all(event <= acceptance)

def HMC(func, initial_position,path_len=.1, step_size=0.1, epsilon = 0.01, epochs=10):
    dimensions = len(initial_position)
    # setup
    steps = int(path_len/step_size) # path_len and step_size are tricky parameters to tune...
    momentum_dist = st.norm(0, 1)
    samples = [initial_position]
    trajectories = []
    success = [] 
    # generate samples
    for e in tqdm(range(epochs)):
        traj = []
        q0 = np.copy(samples[-1])   #np array N-dim
        q1 = np.copy(q0)
        p0 = momentum_dist.rvs(size=dimensions)  #np array N-dim       
        p1 = np.copy(p0) 
        update_1 = np.zeros(2)
        update_2 = np.zeros(2)
        traj.append(np.concatenate([q1, p1, update_1, update_2]))


        # leapfrog integration 
        for s in range(steps):
            q1, p1, A_1, update_1, update_2 = leapfrog_step(q1, p1, step_size, epsilon, func)
            traj.append(np.concatenate([q1, p1, update_1, update_2]))
            

        #flip momentum for reversibility 
        p1 = -1*p1     
        trajectories.append(traj)

        #metropolis acceptance
        accepted = metropolis_acceptance(func, q0, p0, q1, p1)

        #Decide acceptance or refusal
        if accepted:
            samples.append(q1)
            success.append(True)
        else:
            samples.append(q0)
            success.append(False)
    return np.array(samples[1:]), np.array(trajectories), np.array(success)
    
def plot_trajectories(func, trajectories, success, save):
    fig = plt.figure()
    x = np.zeros((50, 50))
    for i in range(50):
        for j in range(50):
            x[i, j] = func([i/50,j/50]) 
    im = plt.imshow(x, extent = [0,1,0,1], origin = 'lower')
    plt.colorbar(im)
    
    for i in range(len(trajectories)):
        if success[i] == True:
            c = 'Green'
        else:
            c = 'Black' 
        plt.scatter(trajectories[i, :,0], trajectories[i, :,1], s = 2, color = c)
        begin = plt.scatter(trajectories[i, 0 ,0], trajectories[i, 0, 1], marker = 's', color = c)
        if len(trajectories)<50:
        	enumeration = plt.annotate(str(i), (trajectories[i, 0 ,0], trajectories[i, 0, 1] + 0.02), color = c)
        end = plt.scatter(trajectories[i, -1, 0], trajectories[i, -1, 1], marker = '*', color = c)
    
    if save:
    	plt.savefig('gra.png')
    plt.show()
    
    