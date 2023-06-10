import modules.metodos_numericos as edo
import modules.odes as odes
import numpy as np
from math import *

import os

def read_data(path) -> list:
    data = []
    # Read Params
    cwd = os.getcwd()

    with open(cwd + "/Prey_Depredator_HollType2/Simulations/S1.txt", "r") as file:
        for line in file.readlines():
            if(line != "\n"):
                data.append([float(x) for x in line.split()])

    return data


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
