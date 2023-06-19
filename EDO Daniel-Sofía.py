# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 05:06:01 2022

@author: Sofia
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

#Parámetros del sistema (positivos)
r=0.82
alpha=1.56
beta=0.87
rho=1.38
m=0.13
mu=0.11
eta=2.41
alpha1=1.12
eta1=1.83
k=12

#Condiciones iniciales del sistema
# Condiciones iniciales del sistema
xpob_inicial = 4  # población inicial de presas jóvenes
ypob_inicial = 2  # población inicial de presas adultas
zpob_inicial = 6  # población inicial de depredadores
duracion = 2  # tiempo de estudio de las poblaciones
cortes = 3  # candidad de valores requeridos por unidad de tiempo

def f1(x,y,z):
    return x*(r*(k-x)/k-beta-alpha*z)

def f2(x,y,z):
    return beta*x-y*((eta*z)/(y+m)-mu)

def f3(x,y,z):
    return z*(alpha1*x+rho*z-(eta1*y)/(y+m))

def Runge_Kutta(stop,N,c1,c2,c3):
    h = 1/N
    t = np.linspace(0,stop,stop*N+1)
    x = np.zeros(stop*N+1)
    y = np.zeros(stop*N+1)
    z = np.zeros(stop*N+1)
    x[0] = c1
    y[0] = c2
    z[0] = c3
    for i in range(stop*N):
        k1 = h*f1(x[i],y[i],z[i])
        l1 = h*f2(x[i],y[i],z[i])
        m1 = h*f3(x[i],y[i],z[i])
        #
        k2 = h*f1(x[i]+k1/2,y[i]+l1/2,z[i]+m1/2)
        l2 = h*f2(x[i]+k1/2,y[i]+l1/2,z[i]+m1/2)
        m2 = h*f3(x[i]+k1/2,y[i]+l1/2,z[i]+m1/2)
        #
        k3 = h*f1(x[i]+k2/2,y[i]+l2/2,z[i]+m2/2)
        l3 = h*f2(x[i]+k2/2,y[i]+l2/2,z[i]+m2/2)
        m3 = h*f3(x[i]+k2/2,y[i]+l2/2,z[i]+m2/2)
        #
        k4 = h*f1(x[i]+k3,y[i]+l3,z[i]+m3)
        l4 = h*f2(x[i]+k3,y[i]+l3,z[i]+m3)
        m4 = h*f3(x[i]+k3,y[i]+l3,z[i]+m3)
        #
        x[i+1]=max(0,x[i]+(k1+2*k2+2*k3+k4)/6)
        y[i+1]=max(0,y[i]+(l1+2*l2+2*l3+l4)/6)
        z[i+1]=max(0,z[i]+(m1+2*m2+2*m3+m4)/6)
    return t,x,y,z

#=================================================================================
#=================================================================================
#Solución del sistema
S = Runge_Kutta(duracion,cortes,xpob_inicial,ypob_inicial,zpob_inicial)
t_discreto = S[0]
presas_joven = S[1]
presas_adult = S[2]
depredadores = S[3]

for i in range(len(presas_joven)):
    print("{0} {1} {2}".format(presas_joven[i], presas_adult[i], depredadores[i]))

#print(S)
#=================================================================================
#=================================================================================

#Gráficos
#Leyendas genéricas
colores=['#9B2D1A','#606099','#000099']
texto=['Depredadores','Presas jóvenes','Presas adultas']
leyenda=[]
for i in range(3):
    leyenda.append(mpatches.Patch(color=colores[i],label=texto[i]))
leyenda1=[]
leyenda1.append(mpatches.Patch(color=colores[0],label=texto[0]))  
leyenda2=[]
leyenda2.append(mpatches.Patch(color=colores[1],label=texto[1])) 
leyenda3=[]
leyenda3.append(mpatches.Patch(color=colores[2],label=texto[2]))  

#Gráfico de variación de los ángulo
fig0 = plt.figure(figsize =(12,6))
ax0 = fig0.add_subplot(111)
bx0 = fig0.add_subplot(111)
cx0 = fig0.add_subplot(111)
ax0.plot(t_discreto,presas_joven,color='#606099')
bx0.plot(t_discreto,presas_adult,color='#000099')
cx0.plot(t_discreto,depredadores,color='#9B2D1A')
plt.ylabel('Poblaciones')
plt.xlabel('Tiempo')       
plt.legend(leyenda,handles=leyenda,loc='upper left')
#fig0.savefig('angulos.jpg')

##Gráfico de trayectoria de los péndulos
#fig1 = plt.figure(figsize =(12,6))
#ax1 = fig1.add_subplot(121)
#bx1 = fig1.add_subplot(122)
#ax1.set_xlim((-l-1,l+1))
#ax1.set_ylim((-l-1,l+1))
#bx1.set_xlim((-l-1,l+1))
#bx1.set_ylim((-l-1,l+1))
#ax1.plot(l*np.sin(angulo_pendulo1),-l*np.cos(angulo_pendulo1),color='#9B2D1A',alpha=0.7,label='Péndulo 1')
#ax1.legend(leyenda1,handles=leyenda1,loc='upper left')
#bx1.plot(l*np.sin(angulo_pendulo2),-l*np.cos(angulo_pendulo2),color='#000099',alpha=0.7,label='Péndulo 2')
#bx1.legend(leyenda2,handles=leyenda2,loc='upper left')
##fig1.savefig('trayectorias.jpg')
#
##=================================================================================
##Gráfico animado de posición de los péndulos (separados)
#ejea2 = -l*np.cos(angulo_pendulo1)
#ejea1 = l*np.sin(angulo_pendulo1)
#ejeb2 = -l*np.cos(angulo_pendulo2)
#ejeb1 = l*np.sin(angulo_pendulo2)
#momentos = t_discreto
#
#fig2 = plt.figure(figsize =(12,6))
#ax2 = fig2.add_subplot(121)
#bx2 = fig2.add_subplot(122)
#
#def update_plot(i):
#    momento = momentos[i]
#    ax2.clear()
#    ax2.set_xlim((-l-1,l+1))
#    ax2.set_ylim((-l-1,l+1))
#    ax2.plot([0,ejea1[i]],[0,ejea2[i]],marker='',linestyle='-',color='black',alpha=0.5,zorder=1)
#    ax2.scatter([0,ejea1[i]],[0,ejea2[i]],s=[50,m1],c=('black','#9B2D1A'),zorder=2)
#    ax2.legend(leyenda1,handles=leyenda1,loc='upper left')
#    bx2.clear()
#    bx2.set_xlim((-l-1,l+1))
#    bx2.set_ylim((-l-1,l+1))
#    bx2.plot([0,ejeb1[i]],[0,ejeb2[i]],marker='',linestyle='-',color='black',alpha=0.5,zorder=1)
#    bx2.scatter([0,ejeb1[i]],[0,ejeb2[i]],s=[50,m2],c=('black','#000099'),zorder=2)
#    bx2.legend(leyenda2,handles=leyenda2,loc='upper left')
#
##anim = animation.FuncAnimation(fig2, update_plot, frames = len(momentos))
##anim.save('pendulo_sep.gif',fps=cortes)
#
##=================================================================================
##Gráfico animado de posición de los péndulos (unidos, incluye representación del resorte)
#fig3 = plt.figure(figsize =(12,12))
#ax3 = fig3.add_subplot(111)
#bx3 = fig3.add_subplot(111)
#cx3 = fig3.add_subplot(111)
#dx3 = fig3.add_subplot(111)
#
#def estiramiento(x,tol):
#    if x<tol:
#        return estiramiento(tol,tol)
#    if x>=tol:
#        return log(1+1/x)
#
#def muelle(n,p1,p2,q1,q2):
#    long = sqrt((p2-q2)**2+(p1-q1)**2)
#    p = (q2-p2)/(q1-p1)
#    muellex = np.zeros(2*n+1)
#    muelley = np.zeros(2*n+1)
#    for i in range(2*n+1):
#        muellex[i] = p1+((q1-p1)*i)/(2*n)
#        muelley[i] = p*(muellex[i]-p1)+p2+(0.175*estiramiento(long,0.75))*sin(pi*((2*i+1)/2))
#    return muellex,muelley
#
#def update_plot(i):
#    momento = momentos[i]
#    ax3.clear()
#    ax3.set_xlim((-l-1,l+1))
#    ax3.set_ylim((-l-1,l+1))
#    bx3.clear()
#    bx3.set_xlim((-l-1,l+1))
#    bx3.set_ylim((-l-1,l+1))
#    cx3.clear()
#    cx3.set_xlim((-l-1,l+1))
#    cx3.set_ylim((-l-1,l+1))
#    dx3.clear()
#    dx3.set_xlim((-l-1,l+1))
#    dx3.set_ylim((-l-1,l+1))
#  
#    dx3.plot([-l+1,l-1],[0,0],marker='',linestyle='--',color='purple',alpha=0.8,zorder=1)
#    bx3.plot([0,ejeb1[i]],[0,ejeb2[i]],marker='',linestyle='-',color='black',alpha=0.3,zorder=1)
#    ax3.plot([0,ejea1[i]],[0,ejea2[i]],marker='',linestyle='-',color='black',alpha=0.3,zorder=1)
#    cx3.plot(muelle(8,ejea1[i],ejea2[i],ejeb1[i],ejeb2[i])[0],muelle(8,ejea1[i],ejea2[i],ejeb1[i],ejeb2[i])[1],marker='',linestyle='-',color='black',alpha=0.8,zorder=1)
#    
#    bx3.scatter([0,ejeb1[i]],[0,ejeb2[i]],s=[100,2*m2],c=('black','#000064'),zorder=2)
#    ax3.scatter([0,ejea1[i]],[0,ejea2[i]],s=[100,2*m1],c=('black','#AE212F'),zorder=2)
#
#    plt.xlabel('time=%.2f s' % momento,fontsize=16)
#    plt.legend(leyenda,handles=leyenda,loc='upper left')
#                              
##anim = animation.FuncAnimation(fig3, update_plot, frames = len(momentos))
##anim.save('pendulo_junto.gif',fps=cortes)