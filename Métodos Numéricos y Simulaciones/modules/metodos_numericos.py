import numpy as np
# f-> sistema, c-> condiciones iniciales, h-> paso, time-> tiempo de estudio

def euler(f, c, h, time):
    x = [c]
    t = [0]
    n = int(time//h)

    for i in range(n):
        x.append(x[i] + h*f(t, x[i]))
        t.append(t[i]+h)
    
    x = np.array(x)
    return t, x


def runge_kutta(f, c, h, time):
    x = [c]
    t = [0]
    n = int(time//h)
    
    for i in range(n):
        k1 = f(t[i], x[i])
        k2 = f(t[i] + h/2, x[i] + 0.5*h*k1)
        k3 = f(t[i] + h/2, x[i] + 0.5*h*k2)
        k4 = f(t[i] + h, x[i] + h*k3)
        t.append(t[i] + h)
        x.append(x[i] + h/6*(k1 + 2*k2 + 2*k3 + k4))

    x = np.array(x)
    return t, x