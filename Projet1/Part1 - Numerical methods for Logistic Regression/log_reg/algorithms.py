import time
import numpy as np
import scipy.sparse.linalg as spla
from numpy.random import randint
from scipy.sparse.linalg.dsolve import linsolve
from log_reg.operators import l1_prox, l2_prox, norm1, norm2sq
from log_reg.utils import print_end_message, print_start_message, print_progress


##########################################################################
# Unconstrained methods
##########################################################################

def GD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Gradient Descent'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
   
    maxit = parameter['maxit'];
    L = parameter['Lips'];
    alpha = 1/L;
    x = parameter['x0'];
   

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()
        

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        x_next = x - alpha*gradf(x);

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter %  5 ==0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
    info['iter'] = maxit
    print_end_message(method_name, time.time() - tic_start)
    return x, info


# gradient with strong convexity
def GDstr(fx, gradf, parameter) :
    """
    Function:  GDstr(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """

    method_name = 'Gradient Descent with strong convexity'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
   
    
    maxit = parameter['maxit'];
    L = parameter['Lips'];
    mu = parameter['strcnvx'];
    alpha = 2/(L+mu);
    x = parameter['x0'];

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start timer
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        x_next = x - alpha*gradf(x);

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info

# accelerated gradient
def AGD(fx, gradf, parameter):
    """
    Function:  AGD (fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx	  - strong convexity parameter
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
   
    maxit = parameter['maxit'];
    L = parameter['Lips'];
    alpha = 1/L;
    x = parameter['x0'];
    y = x;           ### We inialize y to x
    t = 1;
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        x_next = y - alpha*gradf(y);
        t_next = (1 + np.sqrt(4*(t**2) + 1)/2)
        y_next = x_next + (t - 1)/t_next*(x_next - x);

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))


        # Prepare next iteration
        x = x_next;
        t = t_next;
        y = y_next;

    print_end_message(method_name, time.time() - tic_start)
    return x, info

# accelerated gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    Function:  AGDstr(fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx	  - strong convexity parameter
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    method_name = 'Accelerated Gradient with strong convexity'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
   
    L = parameter['Lips'];
    mu = parameter['strcnvx'];
    alpha = ( np.sqrt(L) - np.sqrt(mu) ) / ( np.sqrt(L) + np.sqrt(mu));
    x = parameter['x0'];
    y = x;
    maxit = parameter['maxit'];
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        
        x_next = y - gradf(y)/L ;
        y_next = x_next + alpha*(x_next - x);

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))


        # Prepare next iteration
        x = x_next
        y = y_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info

# LSGD
def LSGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSGD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent with line-search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    method_name = 'Gradient Descent with line search'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
   
    L_prec = parameter['Lips'];
    x = parameter['x0'];
    maxit = parameter['maxit'];
    

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
    
        
        L = 0.5*L_prec
        i = 0
        d = -gradf(x)
        while (  fx(x+d/((2**i)*L)) > ( fx(x) - (np.linalg.norm(d,2))**2/((2**(i+1))*L) ) ):
            i+=1
        L = (2**i)*L
        alpha = 1/L
        x_next = x - alpha*gradf(x)
        
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next
        L_prec=L
        
    print_end_message(method_name, time.time() - tic_start)
    return x, info

# LSAGD
def LSAGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGD (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with line search'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
   
    L_prec = parameter['Lips'];
    x = parameter['x0'];
    y = x;
    t = 1
    maxit = parameter['maxit'];
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        L = 0.5*L_prec
        i = 0
        d = -gradf(y)
       
        while (  fx(y+d/((2**i)*L)) > ( fx(y) - (np.linalg.norm(d,2))**2/((2**(i+1))*L) ) ):
            i+=1
        L = (2**i)*L
        alpha = 1/L
       
        x_next = y-alpha*gradf(y)
        t_next=(1+np.sqrt(4*(L/(L_prec))*t**2+1))/2
        y_next=x_next+((t-1)/(t_next))*(x_next-x)
        

       

       

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        y = y_next
        L_prec=L
        
    print_end_message(method_name, time.time() - tic_start)
    return x, info


# AGDR
def AGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = AGDR (fx, gradf, parameter)
    Purpose:   Implementation of the AGD with adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with restart'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y, t and find the initial function value (fval).
   

    maxit = parameter['maxit'];
    L = parameter['Lips'];
    alpha = 1/L;
    x = parameter['x0'];
    y = x;
    t = 1;
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        x_next = y - alpha*gradf(y);
        

        if fx(x_next) > fx(x):
            y = x
            x_next = x - alpha*gradf(x);
            t = 1
            
            
        t_next = (1 + np.sqrt(4*(t**2) + 1))/2;
        y_next = x_next + (t - 1)/t_next*(x_next - x);


        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        y = y_next
        t = t_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search and adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with line search + restart'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y, t and find the initial function value (fval).
   
    maxit = parameter['maxit'];
    L_prec = parameter['Lips'];
    L = L_prec
    x = parameter['x0'];
    y = x;
    t = 1;
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       

      
        i = 0
        d = -gradf(y)
        L = 0.5*L_prec
       
        while (  fx(y+d/((2**i)*L)) > ( fx(y) - (np.linalg.norm(d,2))**2/((2**(i+1))*L) ) ):
            i+=1
        L = (2**i)*L
        alpha = 1/L
       
        x_next = y-alpha*gradf(y)
        if fx(x) < fx(x_next):
            y = x
            t = 1
            L = 0.5*L_prec
            i = 0
            d = -gradf(y)
       
            while (  fx(y+d/((2**i)*L)) > ( fx(y) - (np.linalg.norm(d,2))**2/((2**(i+1))*L) ) ):
                i+=1
            L = (2**i)*L
            alpha = 1/L
            x_next = y-alpha*gradf(y)
           
        t_next=(1+np.sqrt(4*(L/(L_prec))*t**2+1))/2
        y_next=x_next+((t-1)/(t_next))*(x_next-x)
            
      
            
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        y = y_next
        L_prec = L
        

    print_end_message(method_name, time.time() - tic_start)
    return x, info

def AdaGrad(fx, gradf, parameter):
    """
    Function:  [x, info] = AdaGrad (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the adaptive gradient method with scalar step-size.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Adaptive Gradient method'
    print_start_message(method_name)
    tic_start = time.time()
    
    
    # Initialize x, B0, alpha, grad (and any other)
   
    
    x = parameter['x0']
    maxit = parameter['maxit'];
    Q_prec = 0
    p = len(x)     #to get p,  len(x), required for the identity matrix
    
    alpha = 1
    delta = 10**(-5)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        

        Q = Q_prec + np.linalg.norm(gradf(x))**2;
        H = (np.sqrt(Q)+delta)*np.identity(p)
        x_next = x - alpha*np.dot(np.linalg.inv(H),gradf(x));
        
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        
        info['fx'][iter] = fx(x)
        

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        Q_prec = Q

    print_end_message(method_name, time.time() - tic_start)
    return x, info

# Newton
def ADAM(fx, gradf, parameter):
    """
    Function:  [x, info] = ADAM (fx, gradf, parameter)
    Purpose:   Implementation of ADAM.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    
    method_name = 'ADAM'
    print_start_message(method_name)
    tic_start = time.time()
    
  
    
    # Initialize x, beta1, beta2, alphs, epsilon (and any other)
   
    maxit = parameter['maxit'];
    x = parameter['x0']
    g = gradf(x)
    beta1 = 0.9
    alpha = 0.1
    beta2 = 0.999
    eps = 10**(-8)
    m = np.zeros(x.shape[0])  
    v = np.zeros(x.shape[0]) 
   
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}


    # Main loop.
    for iter in range(maxit):
        tic = time.time()
        

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
       
        m = m*beta1 + (1-beta1)*g;
        v = beta2*v + (1-beta2)*np.power(g,2);
        m_e = m/(1-beta1**(iter+1))
        v_e = v/(1-beta2**(iter+1))
        H = np.sqrt(v_e) + eps
        x_next = x - alpha*np.divide(m_e,H)
       
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
       
        x = x_next
        g = gradf(x) # I think that there is a typo because it doesnt work at all if I use grad(x_k-1), hence I use grad(x)
        
        
        
    print_end_message(method_name, time.time() - tic_start)
    return x, info


def SGD(fx, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent'
    print_start_message(method_name)
    tic_start = time.time()
    
    # Initialize x and alpha.
   
    
    x = parameter['x0'];
    maxit = parameter['maxit'];

    n = parameter['no0functions'] + 1; #n0functions is defined as n-1 so I add 1 to be consistent with the hape of A
    

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}


    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        
        i = np.random.randint(n)  
        
        #alpha = 1/(2*mu*(iter+1))
        alpha = 1/(iter+1)
        
        x_next = x - gradfsto(x,i)*alpha                 # we use the step 1/(k+1) 

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))
            print(i)

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def SAG(fx, gradfsto, parameter):
    """
    Function:  [x, info] = SAG(fx, gradfsto, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent with averaging'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
   
    
    x = parameter['x0']
    maxit = parameter['maxit'];
    Lmax = parameter['Lmax'];
    n = parameter['no0functions'] + 1;
    v = np.zeros((n,len(x)))
    v_prec = v;
    alpha = 1/(16*Lmax)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        
        i = np.random.randint(n)
        
        sum = 0
        for i_k in range(n):
            if i_k == i:
                v[i_k] = gradfsto(x,i)
            else:
                v[i_k] = v_prec[i_k]
            sum+= v[i_k]
            
        x_next = x - sum*alpha / n    
        

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def SVR(fx, gradf, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent with variance reduction'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
   
    
    x = parameter['x0']
    x_tild = x;         ## corresponds to x_tild_0
    v = gradf(x_tild)
    maxit = parameter['maxit'];
    Lmax = parameter['Lmax'];
    n = parameter['no0functions'] + 1;
    q = int(1000*Lmax);
    gamma = 0.01/Lmax;
    
    
    

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
       
        sum = np.zeros(len(v));
        for l in range(q-1):
            i_l = np.random.randint(n);
            v_l = gradfsto(x_tild, i_l) - gradfsto(x, i_l) + v
            x_tild_next = x_tild - gamma*v_l         #x_tild_next = x~_(l+1)    
            sum+= x_tild_next
            x_tild = x_tild_next
        x_next = sum/q;
        
        

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        v = gradf(x)

    print_end_message(method_name, time.time() - tic_start)
    return x, info



##########################################################################
# Prox
##########################################################################



def ista(fx, gx, gradf, proxg, params):
    """
    Function:  [x, info] = ista(fx, gx, gradf, proxg, parameter)
    Purpose:   Implementation of ISTA.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               prox_Lips  - Lipschitz constant for gradient.
               lambda     - regularization factor in F(x)=f(x)+lambda*g(x).
    :param fx:
    :param gx:
    :param gradf:
    :param proxg:
    :param parameter:
    :return:
    """

    method_name = 'ISTA'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters.
   
    lmbd = params['lambda']
    x0 = params['x0'] 
    
    maxit = params['maxit']
    L = params['prox_Lips'];
    
     
    alpha = 1/L
    
    
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the iterate
        x_k = proxg(x_k - alpha*gradf(x_k), alpha*lmbd) 
        

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
       
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))
            
         
    print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def fista(fx, gx, gradf, proxg, params):
    """
    Function:  [x, info] = fista(fx, gx, gradf, proxg, parameter)
    Purpose:   Implementation of FISTA (with optional restart).
    Parameter: x0            - Initial estimate.
               maxit         - Maximum number of iterations.
               prox_Lips     - Lipschitz constant for gradient.
               lambda        - regularization factor in F(x)=f(x)+lambda*g(x).
               restart_fista - enable restart.
    :param fx:
    :param gx:
    :param gradf:
    :param proxg:
    :param parameter:
    :return:
    """
    
    if params['restart_fista']:
        method_name = 'FISTAR'
    else:
        method_name = 'FISTA'
    print_start_message(method_name)

    tic_start = time.time()

    # Initialize parameters
   
    
    x = params['x0']
    maxit = params['maxit']
    t = 1
    L = params['prox_Lips'];
    alpha = 1/L
    lmbd = params['lambda']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_next = x
    y = x
    
    for k in range(maxit):
        tic = time.time()

        # Update iterate
       
        
        x_next = proxg(y - alpha*gradf(y), alpha*lmbd)
        
        
        
        if params['restart_fista']:
            
            if gradient_scheme_restart_condition(x, x_next, y):
                t = 1
                y = x 
                x_next = proxg(y - alpha*gradf(y), alpha*lmbd)
                
        t_next = (1 + np.sqrt(4*(t**2)+1))/2
        y_next = x_next + (t - 1)/t_next*(x_next - x)
            
            
        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_next) + lmbd * gx(x_next)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_next), gx(x_next))
        
        # Prepare the next iteration
        x = x_next
        y = y_next
        t = t_next
        

    print_end_message(method_name, time.time() - tic_start)
    return x_next, info


def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    """
    Whether to restart
    """
   
    
    if np.dot(y_k - x_k_next, x_k_next - x_k) > 0:   
        return True
    
    return False
    
    
    raise NotImplementedError('Implement the method!')


def prox_sg(fx, gx, gradfsto, proxg, params):
    """
    Function:  [x, info] = prox_sg(fx, gx, gradfsto, proxg, parameter)
    Purpose:   Implementation of ISTA.
    Parameter: x0                - Initial estimate.
               maxit             - Maximum number of iterations.
               prox_Lips         - Lipschitz constant for gradient.
               lambda            - regularization factor in F(x)=f(x)+lambda*g(x).
               no0functions      - number of elements in the finite sum in the objective.
               stepsize_at_k - step size as a function of the iterate k.
    :param fx:
    :param gx:
    :param gradfsto:
    :param proxg:
    :param parameter:
    :return:
    """
    
    method_name = 'PROXSG'
    print_start_message(method_name)

    tic_start = time.time()

    # Initialize parameters
   
    
    maxit = params['maxit']
    x0 = params['x0'];
    n = params['no0functions'] + 1
    lmbd = params['lambda']
    X_avg = x0
    acc = 1
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the average iterate
       
        
        i = np.random.randint(n);      
        alpha = params['stepsize_at_k'](k)
         
        x_k = proxg(x_k - gradfsto(x_k, i)*alpha, alpha*lmbd)
        X_avg = ( X_avg*acc + alpha*x_k)
        acc += alpha
        X_avg = X_avg / acc
         

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(X_avg) + lmbd * gx(X_avg)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(X_avg), gx(X_avg))
        
    print_end_message(method_name, time.time() - tic_start)
    return X_avg, info
