import numpy as np
import matplotlib.pyplot as plt

from utils.qaoa import QAOA

def gradient_descent(G, 
                     eta,
                     max_iter = 400,
                     init_params = [1.5,1.5], 
                     method = 'FD', 
                     verbose = 1):
    depth = 1
    h = .01
    threshold = .001
    np.random.seed(9320)
    #params = np.random.uniform(0, np.pi, 2*depth)
    params = init_params
    dparams = np.zeros(2*depth)
    eta = eta
    
    if method =='RMSProp' or method == 'ADAM' or method =='NADAM':
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        m_t = 0
        v_t = 0
    t = 0
    
    count = 0
    training_params = []
    list_f1 = []
    
    f1, _ = QAOA(G, params[0], params[1])
    training_params.append(list(params) + [f1])
    list_f1.append(f1)
    
    save_params = []
    print('Iter   |  gamma  | beta   |  Energy  |  delta\n')
    while True:
        for i in range(depth):
            variation = np.zeros(2*depth)
            variation[i] = h
            params_plus = params + variation
            params_minus = params - variation
            incr, _ = QAOA(G, params_plus[0], params_plus[1])
            decr, _ = QAOA(G, params_minus[0], params_minus[1])
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
    
            variation[depth + i] = h
            params_plus = params + variation
            params_minus = params - variation
            incr, _ = QAOA(G, params_plus[0], params_plus[1])
            decr, _ = QAOA(G, params_minus[0], params_minus[1])
            Delta_f = incr - decr
            
            if method == 'RMSprop':
                m_t = beta_1*m_t + (1-beta_1)*Delta_f
                dparams[depth + i] = 1/(np.sqrt(m_t + epsilon))
            if method == 'ADAM':
                m_t = beta_1*m_t + (1-beta_1)*Delta_f
                m_t_hat = m_t/(1-beta_1**t)
                v_t = beta_2*v_t + (1-beta_2)*Delta_f**2
                v_t_hat = v_t/(1-beta_2**t)
                if method == 'NADAM':
                    m_t_nadam = m_t_hat + (1-beta_1)*Delta_f/(1 - beta_1**t)
                    dparams[depth +i] = m_t_nadam/(np.sqrt(v_t_hat) + epsilon)
                else:
                    dparams[depth + i] = m_t_hat/(np.sqrt(v_t_hat) + epsilon)
            else:
                dparams[depth + i] = (incr - decr)/(2*h)
        
        params = params - dparams * eta
    
        count+=1
        current_f1, _ = QAOA(G, params[0], params[1])
        training_params.append(list(params) + [current_f1])
        list_f1.append(current_f1)
        
        
        if verbose:
            print(count, training_params[-1], dparams*eta)
        #print(' {}   {:.5f}   {:.5f}   {:.3f}'.format(count, training_betas[-1], training_gammas[-1], list_f1[-1]), flush=True)
      
        if np.sum(np.abs(dparams*eta) < 1e-4) == 2:
            print('Converged bc params are varying in the order 1e-4 < threshold')
            break
        
        if count == max_iter:
            flag = False
            print('STOP AT {} STEPS'.format(max_iter))
            break
    
    return training_params
    
def compare_methods(G, num_params, methods):
    
    points_f1 = np.loadtxt('../data/raw/Grid_search_{}x{}.dat'.format(num_params, num_params))
    poin = np.array(points_f1)
    fig = plt.figure()
    plt.title('Compare methods'.format(methods))
    plt.xticks([0,1,2,3], fontsize = 15)
    plt.yticks([0,1,2,3], fontsize = 15)
    plt.imshow(np.reshape(poin[:,2],(num_params, num_params)), extent = [0, np.pi, np.pi, 0])
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    
    markers = ['o', '*', '^', 'x']
    cmpas = ['seismic','spring','binary', 'winter']
    filename = '../data/processed/'
    
    for i, method in enumerate(methods):
        training_params_plot = np.loadtxt('../data/raw/Training_{}.dat'.format(method))
        plt.scatter(training_params_plot[:200,0],
                    training_params_plot[:200,1], 
                    marker = markers[i],
                    cmap = cmpas[i],
                    c = range(len(training_params_plot[:200])),
                    label = method,
                    s = 40)
        filename += method
        filename +='_'

    plt.scatter(1.5,1.5, marker = '+',c = 'Black', s = 100, label = 'Start' )

    plt.xlabel(r'$\gamma$', fontsize=20)
    plt.ylabel(r'$\beta$', fontsize =20)
    plt.legend()
    
    plt.tight_layout()
    filename += '.pdf'
    plt.savefig(filename)
    plt.show()