import numpy as np
import metodos_num_sistemas_edo as SEDO
import matplotlib.pyplot as plt

# Función dx
def f1(t,x,y):
    return x - x*y

# Función dy
def f2(t,x,y):
    return - 2*y + .5*x*y

# SOlución del sistema
def f3(x,y):
    return .5*x + y - (2*np.log(x) + np.log(y))

# FUnción de asignación de x para Euler Implícito
def f1_implicit(t,x,y,h):
    return x/(h*h-h+h*y+1)

# FUnción de asignación de y para Euler Implícito
def f2_implicit(t,x,y,h):
    return 2*y/(-h*h+h*4-x*h+2)


t0 = 0 #tiempo
x0 = 2 #presa
y0 = 1 #depredador
c = f3(x0,y0)

labels = ['euler', 'euler implícito', 'euler mejorado', 'runge-kutta']

# Genera la solución del sistema para los 4 métodos numéricos
def generar_soluciones(h,n):
    X = []
    Y = []
    x,y = SEDO.euler(f1,f2,0,2,1,h,n)
    X.append(x)
    Y.append(y)
    x,y = SEDO.implied_euler(f1_implicit,f2_implicit,0,2,1,h,n)
    X.append(x)
    Y.append(y)
    x,y = SEDO.enhanced_euler(f1,f2,0,2,1,h,n)    
    X.append(x)
    Y.append(y)
    x,y = SEDO.runge_kutta(f1,f2,0,2,1,h,n)
    X.append(x)
    Y.append(y)
    return X,Y  

# Plotea la familia solución en el espacio
def familia_funciones_soluciones():
    SEDO.graph_R3(f3,0.1,0.1,7,3,xlabel='presas',ylabel='depredadores')

# diagram de Fases del sistema
def diagrama():
    SEDO.diagrama_fase(lambda x,y: f1(None,x,y),lambda x,y: f2(None,x,y),0,0,6,12,10,12,norm=0.5,show=False)
    plt.axvline(4, alpha= 1, linestyle='--', color='grey')
    plt.axhline(1, alpha= 1, linestyle='--', color='grey')
    plt.xlabel('presas')
    plt.ylabel('depredadores')
    plt.show()

# Plotea las soluciones para h=0.1
def resolucion_1():    
    X,Y = generar_soluciones(h=0.1,n=100)
    for i in range(2):        
        SEDO.graph(X[i],Y[i],'presas','depredador',loc='upper right',label=labels[i],show=False)
    plt.title('R2 con $h=0.1$')
    plt.show()

    for i in range(2,4):        
        SEDO.graph(X[i],Y[i],'presas','depredador',loc='upper right',label=labels[i],show=False)
    plt.title('R2 con $h=0.1$')
    plt.show()

# Plotea las soluciones explícitamente para h=0.1
def resolucion_2():    
    X,Y = generar_soluciones(h=0.1,n=100)
    T = np.linspace(t0,t0+0.1*len(X[0]),len(X[0]))

    for i in range(4):        
        SEDO.graph(T,X[i],'tiempo','presas',label=labels[i],show=False)  
    plt.title('Presas con $h=0.1$')      
    plt.show()

    for i in range(4):        
        SEDO.graph(T,Y[i],'tiempo','depredador',label=labels[i],show=False)
    plt.title('Depredadores con $h=0.1$')
    plt.show()

# Plotea las soluciones para h=0.001
def resolucion_3():    
    X,Y = generar_soluciones(h=0.001,n=10000)
    T = np.linspace(t0,t0+0.001*len(X[0]),len(X[0]))

    for i in range(4):        
        SEDO.graph(T,X[i],'tiempo','presas',label=labels[i],show=False)  
    plt.title('Presas con $h=0.001$')
    plt.show()

    for i in range(4):        
        SEDO.graph(T,Y[i],'tiempo','depredador',label=labels[i],show=False)
    plt.title('Depredadores con $h=0.001$')
    plt.show()

# Grafica los errores porducidos por los métodos numéricos
def graficar_errores_1():
    X,Y = generar_soluciones(h=0.1,n=100)
    T = np.linspace(t0,t0+0.1*len(X[0]),len(X[0]))
    X_aprox, Y_aprox = np.array(X), np.array(Y)

    C_approx = f3(X_aprox,Y_aprox)
    E = SEDO.error_abs_const(C_approx,f3(x0,y0))
    for i in range(2):        
        SEDO.graph(T,E[i],'tiempo','error',label=labels[i],show=False)
    plt.title('Error con $h=0.1$')
    plt.show()

    for i in range(2,4):        
        SEDO.graph(T,E[i],'tiempo','error',label=labels[i],show=False)
    plt.title('Error con $h=0.1$')
    plt.show()

    for i in range(3,4):        
        SEDO.graph(T,E[i],'tiempo','error',label=labels[i],show=False)
    plt.title('Error con $h=0.1$')
    plt.show()

# Realiza una animación de la solución
def animacion():
    X,Y = SEDO.runge_kutta(f1,f2,t0,x0,y0,h=0.001,n=10000)
    T = np.linspace(t0,t0+0.001*len(X),len(X))
    SEDO.anim(T,(X,Y),step=100)

#################################################
# A partir de aquí puede probar lo que quiera   #
# Recomendamos ejecutar primero estos ejemplos  #
# y luego si lo desea, cree los suyos           #
#################################################

familia_funciones_soluciones()
diagrama()
resolucion_1()
resolucion_2()
resolucion_3()
graficar_errores_1()
animacion()
