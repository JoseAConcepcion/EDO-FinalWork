import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches


def euler(f1, f2, f3, a, b, N, c1, c2, c3):

    h = (b-a)/N
    x = np.zeros((b-a)*N)
    y = np.zeros(len(x))
    z = np.zeros(len(y))
    x[0] = c1
    y[0] = c2
    z[0] = c3

    for i in range(len(x)-1):
        x[i+1] = x[i] + f1(x[i], y[i], z[i])
        y[i+1] = y[i] + f2(x[i], y[i], z[i])
        z[i+1] = z[i] + f3(x[i], y[i], z[i])

    return x, y, z


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


def f1(x, y, z):
    return x*(r*(k-x)/k-beta-alpha*z)


def f2(x, y, z):
    return beta*x-y*((eta*z)/(y+m)-mu)


def f3(x, y, z):
    return z*(alpha1*x+rho*z-(eta1*y)/(y+m))


# Condiciones iniciales del sistema
xpob_inicial = 6  # población inicial de presas jóvenes
ypob_inicial = 4  # población inicial de presas adultas
zpob_inicial = 2  # población inicial de depredadores
duracion = 2  # tiempo de estudio de las poblaciones
cortes = 5  # candidad de valores requeridos por unidad de tiempo

(x, y, z) = euler(f1, f2, f3, 0, duracion,
                  cortes, xpob_inicial, ypob_inicial, zpob_inicial)

for i in range(len(x)):
    print("{0} {1} {2}".format(x[i], y[i], z[i]))

###############################################################################################
t_discreto = np.arange(0,duracion,1/cortes)
print(t_discreto)
presas_joven = x
presas_adult = y
depredadores = z
import matplotlib.patches as mpatches

# Gráficos
# Leyendas genéricas
colores = ['#9B2D1A', '#606099', '#000099']
texto = ['Depredadores', 'Presas jóvenes', 'Presas adultas']
leyenda = []
for i in range(3):
    leyenda.append(mpatches.Patch(color=colores[i], label=texto[i]))
leyenda1 = []
leyenda1.append(mpatches.Patch(color=colores[0], label=texto[0]))
leyenda2 = []
leyenda2.append(mpatches.Patch(color=colores[1], label=texto[1]))
leyenda3 = []
leyenda3.append(mpatches.Patch(color=colores[2], label=texto[2]))

# Gráfico de variación de los ángulo
fig0 = plt.figure(figsize=(12, 6))
plt.plot(t_discreto, presas_joven, color='#606099')
plt.plot(t_discreto, presas_adult, color='#000099')
plt.plot(t_discreto, depredadores, color='#9B2D1A')
plt.ylabel('Poblaciones')
plt.xlabel('Tiempo')
plt.legend(handles=leyenda, loc='upper left')
plt.show()
# fig0.savefig('angulos.jpg')



# # Define legend patches
# colors = ['#9B2D1A', '#606099', '#000099']
# labels = ['Depredadores', 'Presas jóvenes', 'Presas adultas']
# legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]

# # Create subplots and plot data
# fig, (ax0, bx0, cx0) = plt.subplots(nrows=3, ncols=1, figsize=(12, 6), sharex=True)
# ax0.plot(t_discreto, presas_joven, color='#606099')
# bx0.plot(t_discreto, presas_adult, color='#000099')
# cx0.plot(t_discreto, depredadores, color='#9B2D1A')

# # Set axis labels and legend
# fig.suptitle('Variación de las poblaciones')
# fig.text(0.04, 0.5, 'Poblaciones', va='center', rotation='vertical')
# fig.text(0.5, 0.04, 'Tiempo', ha='center')
# fig.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.04, 0.95))
# import matplotlib.patches as mpatches

# # Define legend patches
# colors = ['#9B2D1A', '#606099', '#000099']
# labels = ['Depredadores', 'Presas jóvenes', 'Presas adultas']
# legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]

# # Create subplots and plot data
# fig, (ax0, bx0, cx0) = plt.subplots(nrows=3, ncols=1, figsize=(12, 6), sharex=True)
# ax0.plot(t_discreto, presas_joven, color='#606099')
# bx0.plot(t_discreto, presas_adult, color='#000099')
# cx0.plot(t_discreto, depredadores, color='#9B2D1A')

# # Set axis labels and legend
# fig.suptitle('Variación de las poblaciones')
# fig.text(0.04, 0.5, 'Poblaciones', va='center', rotation='vertical')
# fig.text(0.5, 0.04, 'Tiempo', ha='center')
# fig.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.04, 0.95))

# # Define legend patches
# colors = ['#9B2D1A', '#606099', '#000099']
# labels = ['Depredadores', 'Presas jóvenes', 'Presas adultas']
# legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]

# # Create subplots and plot data
# fig, (ax0, bx0, cx0) = plt.subplots(nrows=3, ncols=1, figsize=(12, 6), sharex=True)
# ax0.plot(t_discreto, presas_joven, color='#606099')
# bx0.plot(t_discreto, presas_adult, color='#000099')
# cx0.plot(t_discreto, depredadores, color='#9B2D1A')

# # Set axis labels and legend
# fig.suptitle('Variación de las poblaciones')
# fig.text(0.04, 0.5, 'Poblaciones', va='center', rotation='vertical')
# fig.text(0.5, 0.04, 'Tiempo', ha='center')
# fig.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.04, 0.95))
