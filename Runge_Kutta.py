import numpy as np
import scipy as sp

def runge_kutta(f1, f2, f3, a, b, N, c1, c2, c3):

    h = (b-a)/N

    xs = np.arange(a, b, h)
    ys = np.zeros(len(xs))
    zs = np.zeros(len(xs))

    xs[0] = c1
    ys[0] = c2
    zs[0] = c3

    for i in range(len(xs)-1):
        x = xs[i]
        y = ys[i]
        z = zs[i]

        k1 = h*f1(x, y, z)
        l1 = h*f2(x, y, z)
        m1 = h*f3(x, y, z)

        k2 = h*f1(x*+0.5*k1, y*+0.5*l1, z*+0.5*m1)
        l2 = h*f2(x*+0.5*k1, y*+0.5*l1, z*+0.5*m1)
        m2 = h*f3(x*+0.5*k1, y*+0.5*l1, z*+0.5*m1)

        k3 = h*f1(x*+0.5*k2, y*+0.5*l2, z*+0.5*m2)
        l3 = h*f2(x*+0.5*k2, y*+0.5*l2, z*+0.5*m2)
        m3 = h*f3(x*+0.5*k2, y*+0.5*l2, z*+0.5*m2)

        k4 = h*f1(x+k3, y+l3, z+m3)
        l4 = h*f1(x+k3, y+l3, z+m3)
        m4 = h*f1(x+k3, y+l3, z+m3)

        xs[i+1] = y + (1/6)*(k1+2*k2+2*k3+k4)*h
        ys[i+1] = y + (1/6)*(l1+2*l2+2*l3+l4)*h
        zs[i+1] = z + (1/6)*(m1+2*m2+2*m3+m4)*h
    
    return xs, ys, zs

###############################


# Parámetros del sistema (positivos)
r = 0.82
alpha = 1.56
beta = 0.87
rho = 1.38
m = 0.13
mu = 0.11
eta = 2.41
alpha1 = 1.12
eta1 = 1.83
k = 12

xpob_inicial = 1  # población inicial de presas jóvenes
ypob_inicial = 4  # población inicial de presas adultas
zpob_inicial = 6  # población inicial de depredadores
duracion = 2  # tiempo de estudio de las poblaciones
cortes = 3  # candidad de valores requeridos por unidad de tiempo

def f1(x, y, z):
    return x*(r*(1-x/k)-beta-alpha*z)


def f2(x, y, z):    
    return beta*x-y*((eta*z)/(y+m)-mu)


def f3(x, y, z):
    return z*(alpha1*x+rho*z-(eta1*y)/(y+m))


(x, y, z) = runge_kutta(f1, f2, f3, 0, duracion,
                        12, xpob_inicial, ypob_inicial, zpob_inicial)

for i in range(len(x)):
    print("{0} {1} {2}".format(x[i], y[i], z[i]))




