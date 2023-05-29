import numpy as np
from scipy.integrate import RK45

def euler(f1 , f2, f3, a, b, N, c1, c2, c3):
    
    h = (b-a)/N
    xs = np.arange(0,N,h)
    ys = np.zeros(len(xs))
    zs = np.zeros(len(ys))
    xs[0] = c1
    ys[0] = c2
    zs[0] = c3

    for i in range(len(xs)):
        
    
        
    return xs, ys, zs

