import numpy as np
import metodos_num_sistemas_edo as SEDO
import matplotlib.pyplot as plt

def f1_1(t,x,y):
    return x-x*x/1.5-2*x*y/(1+x)

def f1_2(t,x,y):
    return x-x*x/1.8-2*x*y/(1+x)

def f2(t,x,y):
    return -y + 4*x*y/(1+x)

h = 0.1
n = 10000
t0 = 0
x0 = 2 
y0 = 1 

# Pinta los diagramas de fase de los dos sistemas
def diagramas_1():
    
    # Define las funciones derivadas como funciones de R2
    dx_R2_1 = lambda x,y: f1_1(None,x,y)
    dx_R2_2 = lambda x,y: f1_2(None,x,y)
    dy_R2 = lambda x,y: f2(None,x,y)

    SEDO.diagrama_fase(dx_R2_1, dy_R2,0,0,1.2,1,16,12,norm=0.07,show=False)   
    plt.axvline(1/3, alpha= 1, linestyle='--', color='grey')
    plt.axhline(7/13.5, alpha= 1, linestyle='--', color='grey')
    plt.title('Digrama con k=1.5')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.25)
    plt.show()

    SEDO.diagrama_fase(dx_R2_2, dy_R2,0,0,1.4,1,16,11,norm=0.07,show=False)  
    plt.axvline(1/3, alpha= 1, linestyle='--', color='grey')
    plt.axhline(8.8/16.2, alpha= 1, linestyle='--', color='grey')
    plt.title('Diagrama con k=1.8')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.45)
    plt.show()

# Pinta los diagramas de fase de los dos sistemas con una curva aproximada
def diagramas_2():
    
    # Define las funciones derivadas como funciones de R2
    dx_R2_1 = lambda x,y: f1_1(None,x,y)
    dx_R2_2 = lambda x,y: f1_2(None,x,y)
    dy_R2 = lambda x,y: f2(None,x,y)

    # Crea el diagrama de fase con vectores de norma 0.05, y de color gris
    SEDO.diagrama_fase(dx_R2_1, dy_R2,0,0,1.2,1,16,12,norm=0.07,show=False,color='grey')    
    x, y = SEDO.runge_kutta(f1_1,f2,0,0.6,0.16,0.1,10000)
    SEDO.graph(x,y,label='$x_0=0.6$ $y_0=0.16$',show=False,loc='upper right')
    plt.axvline(1/3, alpha= 1, linestyle='--', color='black')
    plt.axhline(7/13.5, alpha= 1, linestyle='--', color='black')
    plt.title('Diagrama con k=1.5, con curva')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.25)    
    plt.show()

    SEDO.diagrama_fase(dx_R2_2, dy_R2,0,0,1.4,1,16,11,norm=0.07,show=False,color='grey')    
    x, y = SEDO.runge_kutta(f1_2,f2,0,0.6,0.16,0.1,10000)
    SEDO.graph(x,y,label='$x_0=0.6$ $y_0=0.16$',show=False,loc='upper right')
    plt.axvline(1/3, alpha= 1, linestyle='--', color='black')
    plt.axhline(8.8/16.2, alpha= 1, linestyle='--', color='black')
    plt.title('Diagrama con k=1.8, con curva')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.45)    
    plt.show()

# Plotea la soluciones para h=0.001
def grafica_1():

    x, y = SEDO.runge_kutta(f1_1,f2,0,x0,y0,0.001,500000)
    SEDO.graph(x,y,label='runge-kutta',show=False,loc='upper right')
    plt.title('Solución con k=1.5 y h=0.001')
    plt.axvline(1/3, alpha= 0.5, linestyle='--', color='black')
    plt.axhline(7/13.5, alpha= 0.5, linestyle='--', color='black') 
    plt.show()

    x, y = SEDO.runge_kutta(f1_2,f2,0,x0,y0,0.001,500000)
    SEDO.graph(x,y,label='runge-kutta',show=False,loc='upper right')
    plt.title('Solución con k=1.8 y h=0.001')
    plt.axvline(1/3, alpha= 0.5, linestyle='--', color='black')
    plt.axhline(8.8/16.2, alpha= 0.5, linestyle='--', color='black')
    plt.show()

# Plotea la soluciones explícitamente para h=0.001
def grafica_2():
    
    x, y = SEDO.runge_kutta(f1_1,f2,0,x0,y0,0.001,300000)
    t = np.linspace(t0,t0+0.1*len(x),len(x))
    SEDO.graph(t,x,label='$x=x(t)$',show=False,loc='upper right')
    plt.title('Solución x(t) con k=1.5 y h=0.001')
    plt.axhline(1/3, alpha= 0.5, linestyle='--', color='black') 
    plt.show()

    SEDO.graph(t,y,label='$y=y(t)$',show=False,loc='upper right')
    plt.title('Solución y(t) con k=1.5 y h=0.001')
    plt.axhline(7/13.5, alpha= 0.5, linestyle='--', color='black')
    plt.show()

    SEDO.graph(t,x,label='$x=x(t)$',show=False,loc='upper right')
    plt.title('Solución x(t) y y(t) con k=1.5 y h=0.001')
    SEDO.graph(t,y,label='$y=y(t)$',loc='upper right')

# Plotea la soluciones para h=0.001
def grafica_3():

    x, y = SEDO.runge_kutta(f1_2,f2,0,x0,y0,0.001,300000)
    t = np.linspace(t0,t0+0.1*len(x),len(x))
    SEDO.graph(t,x,label='$x=x(t)$',show=False,loc='upper right')
    plt.title('Solución x(t) con k=1.8 y h=0.001')
    plt.axhline(1/3, alpha= 0.5, linestyle='--', color='black')
    plt.axhline(0.72, alpha= 0.5, linestyle='--', color='red') 
    plt.axhline(0.12, alpha= 0.5, linestyle='--', color='red') 
    plt.show()

    SEDO.graph(t,y,label='$y=y(t)$',show=False,loc='upper right')
    plt.title('Solución y(t) con k=1.8 y h=0.001')
    plt.axhline(8.8/16.2, alpha= 0.5, linestyle='--', color='black')
    plt.axhline(1.13, alpha= 0.5, linestyle='--', color='red') 
    plt.axhline(0.18, alpha= 0.5, linestyle='--', color='red') 
    plt.show()

    SEDO.graph(t,x,label='$x=x(t)$',show=False,loc='upper right')
    plt.title('Solución x(t) y y(t) con k=1.8 y h=0.001')
    SEDO.graph(t,y,label='$y=y(t)$',loc='upper right')

# Animación de x y y para k=1.5
def animacion_1():
    x, y = SEDO.runge_kutta(f1_1,f2,0,x0,y0,0.001,300000)
    t = np.linspace(t0,t0+0.1*len(x),len(x))
    SEDO.anim(t,(x,y),step=1000)
    plt.show()

# Animación de x y y para k=1.8
def animacion_2():
    x, y = SEDO.runge_kutta(f1_2,f2,0,x0,y0,0.001,300000)
    t = np.linspace(t0,t0+0.1*len(x),len(x))
    SEDO.anim(t,(x,y),step=1000)
    plt.show()

#################################################
# A partir de aquí puede probar lo que quiera   #
# Recomendamos ejecutar primero estos ejemplos  #
# y luego si lo desea, cree los suyos           #
#################################################

diagramas_1()
diagramas_2()
grafica_1()
grafica_2()
grafica_3()
animacion_1()
animacion_2()