import modules.metodos_numericos as edo
from modules.odes import *
from modules.plot import *
import numpy as np
from math import *
import glob
import os

h = 0.01
n = 1000

def read_data(folder_path: str, extension: str) -> list:
    data = []
    files = glob.glob(folder_path + "*.{}".format(extension))

    # Read
    for file in files:
        current_data = []
        with open(file, "r") as f:
            for line in f.readlines():
                if(line != "\n"):
                    current_data.append([float(x) for x in line.split()])
        if len(current_data) != 0:
            data.append(current_data)

    return data

def main() -> None:
    data = read_data(
        os.getcwd() + "/Prey_Depredator_HollType2/Simulations/", "txt")

    for simulation in data:
        simulation_data = []
        system = prey_depredator_hollingTypeII(simulation[0])
       
        def f(t,v):
            x,y,z = v
            return np.array([system.f1(t,x,y,z), system.f2(t,x,y,z), system.f3(t,x,y,z)])
       
        for icondition in simulation[1:]:
            euler = edo.euler(f, np.array([icondition[0], icondition[1], icondition[2]]), h, n)
            runge_kutta = edo.runge_kutta(f, np.array([icondition[0], icondition[1], icondition[2]]), h, n)
            simulation_data.append([euler, runge_kutta])
        
        # Plotting Simulations
        title = str(system)


    return


def print_jacobian_symbolic() -> None:

    jacobian = odes.prey_depredator_hollingTypeII.jacobi_matrix_symbolic
    print('- - '*30)  # separador
    representation = "\n"
    for i in range(3):
        for j in range(3):
            representation += str(jacobian[i*3+j]) + '  '*8
        representation += '\n'

    print(representation + '\n')  # para visualizar la matriz

    return


main()
