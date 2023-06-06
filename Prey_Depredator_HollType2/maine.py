import modules.metodos_numericos as edo
import modules.odes as odes
import os
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def main() -> None:

    data = []
    # Read Params
    cwd = os.getcwd()

    with open(cwd + "/Prey_Depredator_HollType2/Simulations/S1.txt", "r") as file:
        for line in file.readlines(-1):
            if(line != "\n"):
                data.append([float(x) for x in line.split()])

    s1 = odes.prey_depredator_hollingTypeII(data[0])
    c = data[1]
    
    def sistem(vector , time):
        x,y,z = vector
        t = time
        return s1.f1(t,x,y,z), s1.f2(t,x,y,z), s1.f3(t,x,y,z)    
    
    t = np.arange(0,100,0.1)
    sol = odeint(sistem, c, t)
    plt.plot(t, sol[:,0], label = 'presas1')
    plt.plot(t, sol[:,1], label = 'presas2')
    plt.plot(t, sol[:,2], label = 'depredadores')
    plt.legend()
    plt.show()

    return 0



main()
