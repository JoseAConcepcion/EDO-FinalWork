import numpy as np
import matplotlib.pyplot as plt
from time import time
from matplotlib.animation import FuncAnimation


def euler(f, g, t, x, y, h, n):
    '''
Método de Euler:

Para resolver sistemas de 2 ecuaciones diferenciales, resueltas para las
deriadas de las variables dependientes.

Nota: El término 'resuelta' se empleará para indicar que está despejada
    para la vriable que se indique.

Parámetros:

f: Función de R3, resuelta para la derivada x

g: Función de R3, resuelta para la derivada y

t: valor inical de la variable independiente t

x: Valor inical de la variable dependinte x

y: Valor inical de la variable dependinte y

h: Tamaño de valor de paso

n: Cantidad de iteraciones
'''
    # Se agreagan los dos primeros valores iniciales de x y y
    X = [x]
    Y = [y]
    # Se realizan n iteraciones para generar los valores de x y y
    for i in range(n):
        # Asignación simultánea en forma de tupla para ahorrar memoria
        # sin declarar explícitamente variables temporales
        x, y, t = x + h*f(t,x,y), y + h*g(t,x,y), t+h
        X.append(x)
        Y.append(y)
    return X,Y


def enhanced_euler(f, g, t, x, y, h, n):
    '''
Método de Euler Mejorado:

Para resolver sistemas de 2 ecuaciones diferenciales, resueltas para las
deriadas de las variables dependientes.
Una pequeña mejora del método de Euler, que aproxima mejor que el método base.

Parámetros:

f: Función de R3, resuelta para la derivada x

g: Función de R3, resuelta para la derivada y

t: valor inical de la variable independiente t

x: Valor inical de la variable dependinte x

y: Valor inical de la variable dependinte y

h: Tamaño de valor de paso

n: Cantidad de iteraciones
'''
    # Se agreagan los dos primeros valores iniciales de x y y
    X = [x]
    Y = [y]
    # Se realizan n iteraciones para generar los valores de x y y
    for i in range(n):
        u = x + h*f(t,x,y)
        v = y + h*g(t,x,y)
        # Asignación simultánea en forma de tupla para ahorrar memoria
        # sin declarar explícitamente variables temporales
        x, y, t = x + .5*h*(f(t,x,y) + f(t+h,u,v)), y + .5*h*(g(t,x,y) + g(t+h,u,v)), t+h
        X.append(x)
        Y.append(y)
    return X,Y


def implied_euler(f_implicit, g_implicit, t, x, y, h, n):
    '''
    Método de Euler Implícito:

    Para resolver sistemas de 2 ecuaciones diferenciales, resueltas para las
    deriadas de las variables dependientes.
    Basados en el método de Euler, pero haceindo un llamado recursivo en cada función,
    por lo que se debe despejar la nueva variable para obtener una asignación NO recursiva.

    Parámetros:

    f_implicit: Función de R3, que sea la función implícita de dx, despejada para la nueva x

    g_implicit: Función de R3, que sea la función implícita de dy, despejada para la nueva y

    t: valor inical de la variable independiente t

    x: Valor inical de la variable dependinte x

    y: Valor inical de la variable dependinte y

    h: Tamaño de valor de paso
        
    n: Cantidad de iteraciones
    '''
    # Se agreagan los dos primeros valores iniciales de x y y
    X = [x]
    Y = [y]
    # Se realizan n iteraciones para generar los valores de x y y
    for i in range(n):
        # Asignación simultánea en forma de tupla para ahorrar memoria
        # sin declarar explícitamente variables temporales
        x, y, t = f_implicit(t,x,y,h), g_implicit(t,x,y,h), t+h
        X.append(x)
        Y.append(y)
    return X,Y


def runge_kutta(f, g, t, x, y, h, n):
    '''
    Método de Runge-Kutta-4

    Para resolver sistemas de 2 ecuaciones diferenciales, resueltas para las
    deriadas de las variables dependientes.
    De los cuatros métodos numéricos implementados, este es el que tiene mayor eficacia numérica
    debido al nivel de correctitud que sucede en el mismo.

    Parámetros:

    f: Función de R3, resuelta para la derivada x

    g: Función de R3, resuelta para la derivada y

    t: valor inical de la variable independiente t

    x: Valor inical de la variable dependinte x

    y: Valor inical de la variable dependinte y

    h: Tamaño de valor de paso

    n: Cantidad de iteraciones
    '''
    # Se agreagan los dos primeros valores iniciales de x y y
    X = [x]
    Y = [y]
    # Se realizan n iteraciones para generar los valores de x y y
    for i in range(n):
        # Creación de los valores de auto-corrección del método
        F1, G1 = f(t,x,y) , g(t,x,y)
        F2, G2 = f(t+h/2, x+.5*h*F1, y+.5*h*G1) , g(t+h/2, x+.5*h*F1, y+.5*h*G1)
        F3, G3 = f(t+h/2, x+.5*h*F2, y+.5*h*G2) , g(t+h/2, x+.5*h*F2, y+.5*h*G2)
        F4, G4 = f(t+h, x+h*F3, y+h*G3) , g(t+h, x+h*F3, y+h*G3)

        # Asignación simultánea en forma de tupla para ahorrar memoria
        # sin declarar explícitamente variables temporales
        t, x, y = t+h, x + h*(F1+2*F2+2*F3+F4)/6, y + h*(G1+2*G2+2*G3+G4)/6
        X.append(x)
        Y.append(y)
    return X, Y



def graph(x, y, xlabel='x', ylabel='y', label=None, loc = 'upper left', show = True):
    '''
    Función para graficar una o varias funciones de R1 a R1, en un mismo intervalo.

    Parámetros:

    x : Lista de la variable independiente. por lo general un intervalo.

    y : Lista con el valor que le corresponde a la función evaluada en el valor de x correspondiente.

    xlabel: Nombre del eje X

    ylabel: Nombre del eje Y

    label: Nombre que se le dará a la función

    loc: Posición de la leyenda (Ver parámetro del método plot de pyplot)

    show: Si se desea mostrar la gráfica luego de pintarla
    '''
    # Plotea los valores de x y y
    plt.plot(x,y,label=label)
    # Asigna nombre de ejes, y posición de la leyenda
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    if show:
        plt.show()

# Plotea una funcion de R2
def graph_R3(f, x0, y0, x_end, y_end, xlabel='x', ylabel='y', show = True):
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    ax.clear()
    x = np.linspace(x0,x_end,100)
    y = np.linspace(y0,y_end,100)
    xmesh,ymesh=np.meshgrid(x,y)
    zmesh = f(xmesh,ymesh)
    ax.plot_surface(xmesh,ymesh,zmesh)
    # Asigna nombre de ejes, y posición de la leyenda
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x0,x_end)
    plt.ylim(y0,y_end)
    if show:
        plt.show()



def anim(x, y, step=1, interval=1):
    '''
    Crea una animación de una o varias funciones en un mismo intervalo

    Parámetros:

    x : Lista de la variable independiente. por lo general un intervalo.

    y : Lista o conjjunto de listas con el valor que le corresponde a la función evaluada en el valor de x correspondiente.
    En caso de ser un conjunto de listas, se pintarán todas a la par, en el mismo intervalo.

    step: Indica la cantidad de puntos que se saltan para refrescar la imagen. 
    Esto no quiere decir que no se pinten todos los puntos, sino cada que intervalos se pintan.
    Mientras mayor sea 'step', más rápido será la animación. Por defecto,
    cuando nose indique, se tomará el tamaño del intervalo entre 50.

    interval: Cada cuantos milisegundos se refrescará la imagen. Mientras menor, más rápido.
    Por defecto y como mínimo 1 milisegundo.
    '''
    # Crea el área para graficar
    fig = plt.figure()
    ax = fig.gca()

    # Crea las funciones de refrescación para pintar
    # en caso de ser solo una lista 'y'
    if np.shape(y[0])==():
        def update(i):
            ax.clear()
            ax.plot(x[:i],y[:i])   
    # en caso de ser un conjunto de listas 'y'     
    else:
        def update(i):
            ax.clear()
            for yj in y:
                ax.plot(x[:i], yj[:i])
    
    # Inicializa el número de step en caso de no haber sido indicado
    if step is None:
        step = len(x)/50
    # Se crea la animación y se muestra
    ani = FuncAnimation(fig, update, range(0,len(x),step),interval=interval,repeat=False)
    plt.show()
    

def diagrama_fase(dx, dy, n0, m0, n_end, m_end, n, m, norm = 0.5, show = True, color='black'):
    '''
Crea el diagram de fase de un sstema de dos ecuaciones diferenciales

Parámetros:

dx: Función de dos variables (sin incluir el término independiente), resuelta para la derivada x

dy: Función de dos variables (sin incluir el término independiente), resuelta para la derivada y

n0: Inicio del intervalo del eje X

m0: Inicio del intervalo del eje Y

n_end: Fin del intervalo del eje X

m_end: Fin del intervalo del eje Y

n: cantidad de vectores por el eje X

m: cantidad de vectores por el eje Y

Opcionales:

norm: Norma que se le desea dar a los vectores, una vez normalizados. por defecto 0.5

show: Si se desea mostrar el diagrama.

color: Colores que tendrán los vectores. Por defecto, negros.
'''
    # Crea los n*m puntos de origen de los vectores
    for i in np.linspace(n0,n_end,n):
        for j in np.linspace(m0,m_end,m):
            # Crea el vector en función de los valores de las derivadas
            v = np.array([dx(j,i), dy(j,i)])
            # Normaliza el vector y modifica su tamaño
            vn = norm*v/np.linalg.norm(v)
            # Pinta el vector.
            plt.quiver(j, i, vn[0], vn[1],scale=1, scale_units= 'xy',color=color)    
    if show:
        plt.show()


def error_abs(x, x_aprox):
    '''
Método para calcular el error absoluto

Parámetros:

x: Valor real

x_aprox: Valor aproximado

Retorna: 

error: El error absoluto cometido
'''
    return np.abs(x-x_aprox)


def error_abs_table(x, x_aprox):
    '''
Método para calcular el error absoluto en una lista de valores

Parámetros:

x: Lista de los valores reales

x_aprox: Lista de los valores aproximados

Retorna: 

error: Lista con los errores absolutos cometidos
'''
    error = []
    for value, value_approx in x, x_aprox:
        error.append(error_abs(value,value_approx))        
    return error


def error_abs_const(x, const):
    '''
Método para calcular el error absoluto en una lista de valores,
donde el valor real es siempre constante.

Parámetros:

x: Lista de los valores reales

x_aprox: Lista de los valores aproximados

Retorna: 

error: Lista con los errores absolutos cometidos
'''
    error = []    
    for value in x:
        error.append(error_abs(value,const))        
    return error


def print_table(table, jump=1, decimals=16):
    '''
Método Auxiliar para imprimir en formato de latex una tabla de datos

Parámetros:

table: matriz bidimensional, donde la dim 1 representa la cantidad de columnas,
y la dim 2 representa la cantidad de filas (contrario al convenio tradicional
pero seleccionado de dicha forma por su comodidad con los otros métodos)

jump: indica la cantidad de filas que se saltarán (sin mostrar) para no imprimirlas
decimals: cantidad de cifras a las que se desea redondear los valores
'''
    s = ""
    for i in np.arange(0, len(table[0]),jump):
        s+= "\\hline "
        for j in range(0, len(table)):            
            s += str(np.around(table[j][i],8)) 
            if j < len(table)-1:
                s+= " & "
        s+= " \\\\\n"
    return s