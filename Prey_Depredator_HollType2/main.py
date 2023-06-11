import modules.metodos_numericos as edo
from modules.odes import *
from modules.plot import *
import numpy as np
import scipy.integrate as ode1
from math import *
import glob
import os

h = 0.01
n = 200


def read_data(file_path: str) -> list:
    data = []
    #files = glob.glob(folder_path + "*.{}".format(extension))
    # Read
    with open(file_path, "r") as f:
        for line in f.readlines():
            if(line != "\n"):
                data.append([float(x) for x in line.split()])

    return data


def main() -> None:
    data = read_data(
        os.getcwd() + "/Prey_Depredator_HollType2/Simulations/S1.txt")

    system = prey_depredator_hollingTypeII(data[0])

    def f(t, v):
        x, y , z = v[0], v[1], v[2]
        return np.array([system.f1(t, x, y, z), system.f2(t, x, y, z), system.f3(t, x, y, z)])

    initials_values = data[1]
    sol = edo.runge_kutta(f, np.array(initials_values), h, n)
    t, v = sol
    x, y, z = v[:,0], v[:,1], v[:,2]
    plotting(t,x,y,z, str(system))


    simulation_kutta = []
    simulation_euler = []

    # for initial_value in initials_values:

    #     #euler = edo.euler(f, np.array(initial_value), h, n)
    #     runge_kutta = edo.runge_kutta(f, np.array(initial_value), h, n)

    #     #simulation_euler.append([euler, runge_kutta])
    #     simulation_kutta.append(runge_kutta)

    #     # Plotting Simulations
    #     title = str(system)
    #     t , v = runge_kutta
    #     x, y, z = v[:,0], v[:,1], v[:,2]
    #     plotting(t,x,y,z, title)
        
    #     #animation(t,x,y,z, title)
    # #plot3d_All(simulation_kutta, str(system))
    return


def print_jacobian_symbolic() -> None:

    jacobian = prey_depredator_hollingTypeII.jacobi_matrix_symbolic
    representation = "\n"
    for i in range(3):
        for j in range(3):
            representation += str(jacobian[i*3+j]) + '  '*8
        representation += '\n'

    print(representation + '\n')  # para visualizar la matriz

    return


main()