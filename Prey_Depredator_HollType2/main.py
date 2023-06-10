import modules.metodos_numericos as edo
import modules.odes as odes
import numpy as np
from math import *
import glob
import os


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
