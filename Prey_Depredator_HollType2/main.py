import modules.metodos_numericos as edo
import modules.odes as odes
import os
from math import *


def main() -> None:

    data = []
    # Read Params
    cwd = os.getcwd()

    with open(cwd + "/Prey_Depredator_HollType2/Simulations/S1.txt", "r") as file:
        for line in file.readlines(-1):
            if(line != "\n"):
                data.append([float(x) for x in line.split()])

    s1 = odes.prey_depredator_hollingTypeII(data[0])

    
    iteraciones = 100000
    h = 1e-2
    c1, c2, c3 = data[1]
    
    t, x, y, z = edo.runge_kutta_4(
        s1.f1, s1.f2, s1.f3, 0, c1, c2, c3, h , iteraciones)

    limit = min(iteraciones, floor((iteraciones)/50)) 

    for i in range(limit): # tiempo , x, y, z
            index = i*50
            print(t[index], x[index], y[index], z[index])
    return 0


main()
