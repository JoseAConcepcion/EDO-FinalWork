import numpy as np
from scipy.integrate import RK45

def euler(f , a, b, N, c1, c2):
    
    h = (b-a)/N
    xs = np.arange(0,N,h)
    ys = np.zeros(len(xs))
    zs = np.zeros(len(ys))
    xs = 0

    for i in range(len(xs)):
    
        RK45()
    return

