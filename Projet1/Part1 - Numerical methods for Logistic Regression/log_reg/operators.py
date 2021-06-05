import numpy as np 


def l1_prox(y, weight):
    """Projection onto the l1-ball.
    """
    #### YOUR CODE GOES HERE
    v = np.zeros(len(y))
    for i in range(len(v)):
        v[i] = np.maximum(np.abs(y[i])-weight,0.)*np.sign(y[i]);    
    return v


def l2_prox(y, weight):
    """Projection onto the l2-ball.
    """
    return (1.0 / (weight + 1)) * y


def norm1(x):
    """Returns the l1 norm `x`.
    """
    return np.linalg.norm(x, 1)


def norm2sq(x):
    """Returns the l2 norm squared of `x`.
    """
    return (1.0 / 2) * np.linalg.norm(x) ** 2

#a= [1,6,4,-8,9,-2,0]
#l1_prox(a,5)
