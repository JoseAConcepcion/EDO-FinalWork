import modules.metodos_numericos as edo
import modules.odes as odes
import os
from math import *
import numpy as np
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


def plotting(t: list, x: list, y: list, z: list, tilte="") -> None:
    plt.title(tilte)
    plt.plot(t, x, label='presas jovenes', color='green')
    plt.plot(t, y, label='presas jovenes', color='blue')
    plt.plot(t, z, label='depredadores', color='red')
    plt.legend()
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

    return


def main3() -> None:
    data = read_data()
    s1 = odes.prey_depredator_hollingTypeII(data[0])
    c = data[1]

    def system(time, vector):
        x, y, z = vector
        t = time
        return np.array([s1.f1(t, x, y, z), s1.f2(t, x, y, z), s1.f3(t, x, y, z)])

    t, sol = edo.runge_kutta(system, c, 0.1, 1000)

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
main2() # -> Ejecutar el metodo de Runge-Kutta 4,(python)
# main3() # -> Ejecutar el metodo de Runge-Kutta 4,(vectorizado)
# -> imprimir la matriz jacobiana, true para mostrar los valores propios
#print_jacobian(True)