import math
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style
from prettytable import PrettyTable  # ! sudo pip install prettytable

# Parámetros del sistema (positivos)
r = 0.82
alpha = 1.56
beta = 0.87
rho = 1.38
m = 0.13
mu = 0.11
eta = 2.41
alpha1 = 1.12
eta1 = 1.83
k = 12

# Parámetros de integración
t0 = 0.0
y0 = np.array([4, 5, 3])
h = 0.01
n = 1000

# definicion de la funcion
def f(t, y):
    x, y, z = y
    dxdt = x*(r*(k-x)/k-beta-alpha*z)
    dydt = beta*x-y*((eta*z)/(y+m)-mu)
    dzdt = z*(alpha1*x+rho*z-(eta1*y)/(y+m))
    dxdt = dxdt if x > 0 else 0.1
    dydt = dydt if y > 0 else 0.1
    dzdt = dzdt if z > 0 else 0.1
    if x < 1 or y < 1:
        dzdt = dzdt if dzdt < 0 else -dzdt
    return np.array([dxdt, dydt, dzdt])


def adams_bashforth_system(f, t0, y0, h, n):
    # Inicialización
    t = [t0]
    y = [y0]

    #Euler para obtener los primeros dos valores de la solución
    y_next = y0 + h * np.array(f(t0, y0))
    y.append(y_next)
    t.append(t0 + h)

    # Iteramos utilizando el método de Adams-Bashforth
    for i in range(2, n+1):
        y_prev = y[i-1]
        y_prev_prev = y[i-2]
        k1 = f(t[i-1], y_prev)
        k2 = f(t[i-2], y_prev_prev)
        y_next = y_prev + h*(3*k1 - k2)/2
        y.append(y_next)
        t.append(t[i-1] + h)

    return y


# Aproximación de la solución
y = adams_bashforth_system(f, t0, y0, h, n)
y = np.array(y)
presas_joven = y[:, 0]
presas_adult = y[:, 1]
depredadores = y[:, 2]


# Crear la tabla
tabla = PrettyTable()
tabla.field_names = ["i", "Presas Joven", "Presas Adultas", "Depredadores"]

# Agregar los datos a la tabla
for i in range(len(presas_joven)):
    tabla.add_row([Fore.YELLOW + str(i+1) + Style.RESET_ALL,
                   Fore.BLUE + str(presas_joven[i]) + Style.RESET_ALL,
                   Fore.GREEN + str(presas_adult[i]) + Style.RESET_ALL,
                   Fore.RED + str(depredadores[i]) + Style.RESET_ALL])

# Imprimir la tabla
print(tabla)

# Graficar la solución
plt.plot(y[:, 0], label="x")
plt.plot(y[:, 1], label="y")
plt.plot(y[:, 2], label="z")
plt.legend()
plt.xlabel("t")
plt.ylabel("Variables dependientes")
plt.show()
