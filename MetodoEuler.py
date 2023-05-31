import numpy as np
import matplotlib.pyplot as plt

# Definir las ecuaciones diferenciales
def f(t, y):
    x, y, z = y
    dxdt = r*x*(1-x/k) - beta*x - alpha*x*z
    dydt = beta*x - eta*z*y/(y+m) - mu*y
    dzdt = alpha*x*z + rho*z**2 - eta*z*y/(y+m)
    return [dxdt, dydt, dzdt]

# Definir los parámetros y condiciones iniciales
r = 0.82
k = 12
beta = 0.87
alpha = 1.56
eta = 2.41
mu = 0.11
rho = 1.38
m = 0.13
y0 = [0.3, 2.4, 3.9]

# Definir el tiempo y el tamaño del paso
t0 = 0
tf = 3
h = 0.2
t = np.arange(t0, tf, h)

# Crear un arreglo para almacenar los valores de x, y y z
y = np.zeros((len(t), 3))

# Implementar el método deEuler
y[0, :] = y0
for i in range(len(t)-1):
    y[i+1, :] = y[i, :] + h * np.array(f(t[i], y[i, :]))

# Graficar los resultados
plt.plot(t, y[:, 0], 'b', label='x')
plt.plot(t, y[:, 1], 'g', label='y')
plt.plot(t, y[:, 2], 'r', label='z')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()