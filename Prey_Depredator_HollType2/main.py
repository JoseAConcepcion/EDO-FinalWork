import modules.RK4 as rk4
import modules.odes as odes
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sympy import symbol

def main() -> None:


    data = []
    # Read Params
    cwd = os.getcwd()

    with open(cwd + "/Prey_Depredator_HollType2/Simulations/S6.txt", "r") as file:
        for line in file.readlines(-1):
            if(line != "\n"):
                data.append([float(x) for x in line.split()])

    s1 = odes.Prey_Depredator_HollingTypeII(data[0])

    c1, c2, c3 = data[1]
    t = np.linspace(0, 1000, 1000)
    # t, x, y, z = rk4.runge_kutta_4(
    #     s1.f1, s1.f2, s1.f3, 0, c1, c2, c3, 0.001, 1000000)
    def system(v,t):
        x,y,z = v
        return [s1.f1(t,x,y,z), s1.f2(t,x,y,z), s1.f3(t,x,y,z)]
    sol = odeint(system, [c1,c2,c3], t)

    limit = 50

    # for i in range(limit): # tiempo , x, y, z
    #     print(t[i], x[i], y[i], z[i])


    plt.plot(t, sol[0:,0], label="x")
    plt.plot(t, sol[0:, 1], label="y")
    plt.plot(t, sol[0:, 2], label="z")
    plt.legend()
    plt.show()
    return

def find_initial():
    a = np.array([[2 , -2, 3], [1,1,1], [1,3,-1]])
    b = np.array([1, 2, 3])
    x = np.linalg.solve(a, b)
    print(x)
    return x

def main2():
    c1,c2,c3 = find_initial()
    def f1(t, x,y,z):
        return 2*x-2*y+3*z
    def f2(t, x,y,z):
        return x+y+z
    def f3(t, x,y,z):
        return x+3*y-z

    def system(v , t):
        x,y,z = v
        return [f1(t,x,y,z), f2(t,x,y,z), f3(t,x,y,z)]
    
    t = np.arange((0, 10, 0.01))
    sol = odeint(system, [c1,c2,c3], t)

