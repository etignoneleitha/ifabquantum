import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(acq_func, 
                     l_rate,
                     init_params, 
                     max_iter = 400,
                     method = 'FD', 
                     verbose = 1):
    if not init_params:
        raise ValueError('You have to give the initial params')
    dimension = len(init_params)
    h = .01
    threshold = .001
    np.random.seed(9320)
    #params = np.random.uniform(0, np.pi, 2*depth)
    params = init_params
    dparams = np.zeros(dimension)
    l_rate = l_rate
    
    if method =='RMSProp' or method == 'ADAM' or method =='NADAM':
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        m_t = 0
        v_t = 0
    t = 0
    
    count = 0
    training_params = []
    
    acq = acq_func(params)
    training_params.append(list(params) + [acq])
    
    save_params = []
    print('Iter   |  gamma  | beta   |  Energy  |  delta\n')
    while True:
        for i in range(dimension):
            variation = np.zeros(dimension)
            variation[i] = h
            params_plus = params + variation
            params_minus = params - variation
            incr = acq_func(params_plus)
            decr = acq_func(params_minus)
            Delta_f = incr - decr
            
            t += 1
            
            if method == 'RMSProp':
                m_t = beta_1*m_t + (1-beta_1)*Delta_f
                dparams[i] = 1/(np.sqrt(m_t + epsilon))
            if method == 'ADAM':
                m_t = beta_1*m_t + (1-beta_1)*Delta_f
                m_t_hat = m_t/(1-beta_1**t)
                v_t = beta_2*v_t + (1-beta_2)*Delta_f**2
                v_t_hat = v_t/(1-beta_2**t)
                if method == 'NADAM':
                    m_t_nadam = m_t_hat + (1-beta_1)*Delta_f/(1 - beta_1**t)
                    dparams[i] = m_t_nadam/(np.sqrt(v_t_hat) + epsilon)
                else:
                    dparams[i] = m_t_hat/(np.sqrt(v_t_hat) + epsilon)
            else:
                dparams[i] = (Delta_f)/(2*h)
        
        #GRADIENT ASCENT
        params = params + dparams * l_rate
    
        count+=1
        current_acq = acq_func(params)
        training_params.append(list(params) + [current_acq])
        
        if verbose:
            print(count, training_params[-1], dparams*l_rate)
        #print(' {}   {:.5f}   {:.5f}   {:.3f}'.format(count, training_betas[-1], training_gammas[-1], list_f1[-1]), flush=True)
      
        if np.sum(np.abs(dparams*l_rate) < 1e-4) == 2:
            print('Converged bc params are varying in the order 1e-4 < threshold')
            break
        
        if count == max_iter:
            flag = False
            print('STOP AT {} STEPS'.format(max_iter))
            break
    
    return training_params