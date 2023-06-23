import modules.metodos_numericos as edo
from modules.odes import *
from modules.plot import *
from math import *
import numpy as np
import sympy as sp
from scipy.integrate import odeint  


def S1():  # first simulation

    params = [0.82, 0.87, 1.56, 1.12, 2.41, 1.83, 12, 1.38, 0.13, 0.11]
    system = prey_depredator_hollingTypeII(params)
    
    h = 0.001

    def f(t, v):
        x, y, z = v[0], v[1], v[2]
        return np.array([system.f1(t, x, y, z), system.f2(t, x, y, z), system.f3(t, x, y, z)])

    first_values, t1 = np.array([3.01, 5.05, 4.28]), 3
    second_values, t2 = np.array([4.6,  5.9,  3.1]), 5
    third_values, t3 = np.array([12.2, 22.1, 21.1]), 3

    tx1, sol1 = edo.runge_kutta(f, first_values, h, t1)
    tx2, sol2 = edo.runge_kutta(f, second_values, h, t2)
    #tx3, sol3 = edo.euler(f, third_values, h, t3) # ni con euler
    # tx3 = np.arange(0, t3, 0.00001)
    # sol3 = odeint(f, third_values, tx3, tfirst = True) # problemas de overflow en la tercera simulacion

    title = str(system)
    plotting(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)
    animation(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)

    plotting(tx2, sol2[:, 0], sol2[:, 1], sol2[:, 2], title)
    animation(tx2, sol2[:, 0], sol2[:, 1], sol2[:, 2], title)

    # plotting(tx3, sol3[:, 0], sol3[:, 1], sol3[:, 2], title)
    # animation(tx3, sol3[:, 0], sol3[:, 1], sol3[:, 2], title) # problemas de overflows
    t = 50
    h =0.001
    _,d1 = edo.runge_kutta(f, first_values, h, t)
    _,d2 = edo.runge_kutta(f, second_values, h, t)

    plot3d_All([d1,d2], title)

    return

def S2():  # 2nd simulation

    params = [1.32, 0.87, 1.16, 0.72, 1.6, 0.41, 2.8, 0.78, 0.23, 0.11] 
    system = prey_depredator_hollingTypeII(params)

    h = 0.01

    def f(t, v):
        x, y, z = v[0], v[1], v[2]
        return np.array([system.f1(t, x, y, z), system.f2(t, x, y, z), system.f3(t, x, y, z)])

    first_values, t1 = np.array([0.3, 2.4, 3.9]), 3
    second_values, t2 = np.array([0.6, 2.4, 4.1]), 5
    third_values, t3 = np.array([2.1,1.2,1.1]), 5

    tx1, sol1 = edo.runge_kutta(f, first_values, h, t1)
    tx2, sol2 = edo.runge_kutta(f, second_values, h, t2)
    tx3, sol3 = edo.runge_kutta(f, third_values, h, t3)

    title = str(system)
    plotting(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)
    animation(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)

    plotting(tx2, sol2[:, 0], sol2[:, 1], sol2[:, 2], title)
    animation(tx2, sol2[:, 0], sol2[:, 1], sol2[:, 2], title)

    plotting(tx3, sol3[:, 0], sol3[:, 1], sol3[:, 2], title)
    animation(tx3, sol3[:, 0], sol3[:, 1], sol3[:, 2], title)

    t = 50
    _,d1 = edo.runge_kutta(f, first_values, h, t)
    _,d2 = edo.runge_kutta(f, second_values, h, t)
    _,d3 = edo.runge_kutta(f, third_values, h, t)

    plot3d_All([d1,d2,d3], title)

    return


def S3():  # 3rd simulation

    params = [11.32, 0.87, 0.76, 0.72, 0.6, 0.41, 2.8, 0.78, 0.23, 0.11] 
    system = prey_depredator_hollingTypeII(params)

    h = 0.01

    def f(t, v):
        x, y, z = v[0], v[1], v[2]
        return np.array([system.f1(t, x, y, z), system.f2(t, x, y, z), system.f3(t, x, y, z)])

    first_values, t1 = np.array([0.3, 2.4, 3.9]), 3

    tx1, sol1 = edo.runge_kutta(f, first_values, h, t1)

    title = str(system)
    plotting(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)
    animation(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)

    t = 50
    _,d1 = edo.runge_kutta(f, first_values, h, t)

    plot3d_All([d1], title)

    return

def S4(): 

    params = [1.32, 0.87, 1.16, 0.72, 1.2, 0.41, 2.8, 0.78, 0.23, 0.11] 
    system = prey_depredator_hollingTypeII(params)

    h = 0.01

    def f(t, v):
        x, y, z = v[0], v[1], v[2]
        return np.array([system.f1(t, x, y, z), system.f2(t, x, y, z), system.f3(t, x, y, z)])

    first_values, t1 = np.array([1.2, 2.1, 2.4]), 3

    tx1, sol1 = edo.runge_kutta(f, first_values, h, t1)

    title = str(system)
    plotting(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)
    animation(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)

    t = 50
    _,d1 = edo.runge_kutta(f, first_values, h, t)

    plot3d_All([d1], title)

    return


def S5(): 

    params = [1.32, 0.87, 1.16, 0.72, 0.3095, 0.41, 2.8, 0.78, 0.23, 0.11] 
    system = prey_depredator_hollingTypeII(params)

    h = 0.01

    def f(t, v):
        x, y, z = v[0], v[1], v[2]
        return np.array([system.f1(t, x, y, z), system.f2(t, x, y, z), system.f3(t, x, y, z)])

    first_values, t1 = np.array([0.3, 2.4, 3.9]), 3
    second_values, t2 = np.array([4.1,2.2,5.1]), 5
    third_values, t3 = np.array([2.1,1.2,1.1]), 5

    tx1, sol1 = edo.runge_kutta(f, first_values, h, t1)
    tx2, sol2 = edo.runge_kutta(f, second_values, h, t2)
    tx3, sol3 = edo.runge_kutta(f, third_values, h, t3)

    title = str(system)
    plotting(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)
    animation(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)

    plotting(tx2, sol2[:, 0], sol2[:, 1], sol2[:, 2], title)
    animation(tx2, sol2[:, 0], sol2[:, 1], sol2[:, 2], title)

    plotting(tx3, sol3[:, 0], sol3[:, 1], sol3[:, 2], title)
    animation(tx3, sol3[:, 0], sol3[:, 1], sol3[:, 2], title)

    t = 50
    _,d1 = edo.runge_kutta(f, first_values, h, t)
    _,d2 = edo.runge_kutta(f, second_values, h, t)
    _,d3 = edo.runge_kutta(f, third_values, h, t)

    plot3d_All([d1,d2,d3], title)

    return

def misterius_simulation(): # 3.1.4 n>b and n>a

    params = [0.82, 0.87, 0.76, 1.2, 2.41, 1.83, 2.8, 1.38, 0.13, 0.11]
    system = prey_depredator_hollingTypeII(params)
    
    h = 0.001

    def f(t, v):
        x, y, z = v[0], v[1], v[2]
        return np.array([system.f1(t, x, y, z), system.f2(t, x, y, z), system.f3(t, x, y, z)])

    first_values, t1 = np.array([1.2, 2.1, 2.4]), 3
   
    tx1, sol1 = edo.runge_kutta(f, first_values, h, t1)
    
    title = str(system)
    plotting(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)
    animation(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)

    plot3d_All([sol1], title)

    return

def misterius_simulation2(): # 3.1.5 n = a 

    params = [0.82, 1.3, 1.2, 1.2, 1.2, 1.83, 2.8, 1.38, 0.13, 0.11]
    system = prey_depredator_hollingTypeII(params)
    
    h = 0.001

    def f(t, v):
        x, y, z = v[0], v[1], v[2]
        return np.array([system.f1(t, x, y, z), system.f2(t, x, y, z), system.f3(t, x, y, z)])

    first_values, t1 = np.array([1.2, 2.1, 2.4]), 3
   
    tx1, sol1 = edo.runge_kutta(f, first_values, h, t1)
    
    title = str(system)
    plotting(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)
    animation(tx1, sol1[:, 0], sol1[:, 1], sol1[:, 2], title)

    plot3d_All([sol1], title)

    return

def print_jacobian_symbolic() -> None:

    jacobian = prey_depredator_hollingTypeII.jacobi_matrix_symbolic()
    representation = "\n"
    for i in range(3):
        for j in range(3):
            representation += str(jacobian[i*3+j]) + '  '*8
        representation += '\n'

    print(representation + '\n')  # para visualizar la matriz

    return


S1()
#S2()
#S3()
#S4()
#S5()
#misterius_simulation()
#misterius_simulation2()

#print_jacobian_symbolic()