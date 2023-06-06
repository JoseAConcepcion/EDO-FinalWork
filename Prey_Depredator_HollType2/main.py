import modules.metodos_numericos as edo
import modules.odes as odes
import os
from math import *
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt



def read_data() -> list:
    data = []
    # Read Params
    cwd = os.getcwd()

    with open(cwd + "/Prey_Depredator_HollType2/Simulations/S1.txt", "r") as file:
        for line in file.readlines(-1):
            if(line != "\n"):
                data.append([float(x) for x in line.split()])

    return data

def main() -> None:
    data = read_data()    

    s1 = odes.prey_depredator_hollingTypeII(data[0])
    iteraciones = 1000
    h = 0.1
    c1, c2, c3 = data[1]
    
    t, x, y, z = edo.runge_kutta_4(
        s1.f1, s1.f2, s1.f3, 0, c1, c2, c3, h , iteraciones)

    limit = min(iteraciones, floor((iteraciones)/50)) 

    for i in range(limit): # tiempo , x, y, z
            index = i*50
            print(t[index], x[index], y[index], z[index])

    plt.plot(t, x, label = 'presas1')
    plt.plot(t, y, label = 'presas2')
    plt.plot(t, z, label = 'depredadores')
    plt.legend()
    plt.show()
    return 

def main2() -> None:
    data = read_data()
    s1 = odes.prey_depredator_hollingTypeII(data[0])
    c = data[1]
    
    def system(vector , time):
        x,y,z = vector
        t = time
        return s1.f1(t,x,y,z), s1.f2(t,x,y,z), s1.f3(t,x,y,z)    
    
    t = np.arange(0,100,0.1)
    sol = odeint(system, c, t)
    plt.plot(t, sol[:,0], label = 'presas1')
    plt.plot(t, sol[:,1], label = 'presas2')
    plt.plot(t, sol[:,2], label = 'depredadores')
    plt.legend()
    plt.show()

    return

def main3() -> None:
    data = read_data()
    s1 = odes.prey_depredator_hollingTypeII(data[0])
    c = data[1]
    
    def system(time, vector):
        x,y,z = vector
        t = time
        return np.array([s1.f1(t,x,y,z), s1.f2(t,x,y,z), s1.f3(t,x,y,z)])    
    
    t, sol = edo.runge_kutta(system, c, 0.1, 1000)

    prey1 = []
    prey2 = []
    depredator = []
    for i in range(len(sol)):
        prey1.append(sol[i][0])
        prey2.append(sol[i][1])
        depredator.append(sol[i][2]) 

    plt.plot(t, prey1, label = 'presas1')
    plt.plot(t, prey2, label = 'presas2')
    plt.plot(t, depredator, label = 'depredadores')
    plt.legend()
    plt.show()

    return

main() # -> Ejecutar el metodo de Runge-Kutta 4,(old)
#main2() # -> Ejecutar el metodo de Runge-Kutta 4,(python)
main3()
