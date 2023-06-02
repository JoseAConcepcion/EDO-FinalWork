import modules.RK4 as rk4
import modules.odes as odes
import os
def main() -> None:

    data = []
    # Read Params
    cwd = os.getcwd()
    
    with open( cwd + "/Prey_Depredator_HollType2/Simulations/S1.txt", "r") as file:
        for line in file.readlines(-1):
            if(line != "\n"):
                data.append([float(x) for x in line.split()])

    s1 = odes.Prey_Depredator_HollingTypeII(data[0])
    
    c1, c2, c3 = data[1]
    t, x, y, z = rk4.runge_kutta_4(
        s1.f1, s1.f2, s1.f3, 0, c1, c2, c3, 1e-8, 10000)

    limit = 50

    for i in range(limit):
        print(t[i], x[i], y[i], z[i])

    return


main()
