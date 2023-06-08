import modules.metodos_numericos as edo
import modules.odes as odes
import os
from math import *
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from time import time
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d


def read_data() -> list:
    data = []
    # Read Params
    cwd = os.getcwd()

    with open(cwd + "/Prey_Depredator_HollType2/Simulations/S1.txt", "r") as file:
        for line in file.readlines(-1):
            if(line != "\n"):
                data.append([float(x) for x in line.split()])

    return data


def plotting(t: list, x: list, y: list, z: list, tilte="") -> None:
    plt.title(tilte)
    plt.plot(t, x, label='presas jovenes', color='green')
    plt.plot(t, y, label='presas jovenes', color='brown')
    plt.plot(t, z, label='depredadores', color='red')
    plt.legend()
    plt.show()
    return


def animation(t: list, y1: list, y2: list, y3: list, title="") -> None:

    fig = plt.figure()
    line1, = plt.plot(t, y1, label='presas jovenes', color="green")
    line2, = plt.plot(t, y2, label='presas adultas', color="brown")
    line3, = plt.plot(t, y3, label='depredadores', color="red")
    plt.legend()

    def update(frame):
        y1[:-1] = y1[1:]
        y2[:-1] = y2[1:]
        y3[:-1] = y3[1:]

        line1.set_ydata(y1)
        line2.set_ydata(y2)
        line3.set_ydata(y3)
        return line1, line2, line3

    plt.title(title)
    ani = FuncAnimation(fig, update, frames=len(t), interval=75, repeat=False)
    plt.show()
    return


def dynamic_behaviour(x: list, y: list, z: list) -> None:
    X = np.array(x)
    Y = np.array(y)
    Z = np.array(z)
       
    for i in range(len(x)):
        x[i] = max(0,x[i])
        y[i] = max(0,x[i])
        z[i] = max(0,x[i])        

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('young prey')
    ax.set_ylabel('old prey')
    ax.set_zlabel('predators')

    ax.plot(X, Y, Z, 'gray')
    plt.show()
    return

def main() -> None:
    data = read_data()

    s1 = odes.prey_depredator_hollingTypeII(data[0])
    iteraciones = 1000
    h = 0.1
    c1, c2, c3 = data[1]

    t, x, y, z = edo.runge_kutta_4(
        s1.f1, s1.f2, s1.f3, 0, c1, c2, c3, h, iteraciones)

    limit = min(iteraciones, floor((iteraciones)/50))

    for i in range(limit):  # tiempo , x, y, z
        index = i*50
        print(t[index], x[index], y[index], z[index])

    plotting(x, y, z)

    return


def main2() -> None:
    data = read_data()
    s1 = odes.prey_depredator_hollingTypeII(data[0])
    c = data[1]

    def system(vector, time):
        x, y, z = vector
        t = time
        return s1.f1(t, x, y, z), s1.f2(t, x, y, z), s1.f3(t, x, y, z)

    t = np.arange(0, 2, 0.01)
    sol = odeint(system, c, t)

    plotting(t, sol[:, 0], sol[:, 1], sol[:, 2])
    animation(t, sol[:, 0], sol[:, 1], sol[:, 2] ,str(s1))
    dynamic_behaviour(sol[:, 0], sol[:, 1], sol[:, 2])
    return


def main3() -> None:
    data = read_data()
    s1 = odes.prey_depredator_hollingTypeII(data[0])
    c = data[1]

    def system(time, vector):
        x, y, z = vector
        t = time
        return np.array([s1.f1(t, x, y, z), s1.f2(t, x, y, z), s1.f3(t, x, y, z)])

    t, sol = edo.runge_kutta(system, c, 0.001, 1000)

    prey1 = []
    prey2 = []
    depredator = []
    for i in range(len(sol)):
        prey1.append(sol[i][0])
        prey2.append(sol[i][1])
        depredator.append(sol[i][2])

    plotting(prey1, prey2, depredator)

    return


def print_jacobian(eig=False) -> None:

    eigvalues, jacobian = odes.prey_depredator_hollingTypeII.jacobi_matrix()
    print('- - '*30)  # separador
    representation = ""
    for i in range(3):
        for j in range(3):
            representation += str(jacobian[i*3+j]) + '  '*8
        representation += '\n'

    print(representation)  # para visualizar la matriz
    print('- - '*30)  # separador

    if (eig):
        print('Eigenvalues: ', eigvalues)

    return


# main() # -> Ejecutar el metodo de Runge-Kutta 4,(normal)
main2()  # -> Ejecutar el metodo de Runge-Kutta 4,(python)
# main3() # -> Ejecutar el metodo de Runge-Kutta 4,(vectorizado)
# -> imprimir la matriz jacobiana, true para mostrar los valores propios
# print_jacobian(True)


print('- - '*25)
# Aniumation
