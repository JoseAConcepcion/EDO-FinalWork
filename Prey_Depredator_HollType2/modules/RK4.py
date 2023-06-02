import numpy as np


def runge_kutta_4(
        f1, f2, f3,                               # -> derivadas del sistema evaluables en t,x,y,z
        t: float, x: float, y: float, z: float,   # -> valores iniciales
        h:float, n:int                            # -> tamaño de paso y numero de pasos
        ) -> tuple[list, list, list, list]:       # -> valores de t, x, y, t
   
    xs = [x]
    ys = [y]
    zs = [z]
    ts = [t]

    for i in range(n):
        
        k1, l1, m1 = f1(t, x, y, z), f2(t, x, y, z), f3(t, x, y, z)

        k2 = f1(t + h/2, x + h/2*k1, y + h/2*l1, z + h/2*m1)
        l2 = f2(t + h/2, x + h/2*k1, y + h/2*l1, z + h/2*m1)
        m2 = f3(t + h/2, x + h/2*k1, y + h/2*l1, z + h/2*m1)

        k3 = f1(t + h/2, x + h/2*k2, y + h/2*l2, z + h/2*m2)
        l3 = f2(t + h/2, x + h/2*k2, y + h/2*l2, z + h/2*m2)
        m3 = f3(t + h/2, x + h/2*k2, y + h/2*l2, z + h/2*m2)

        k4 = f1(t + h, x + h*k3, y + h*l3, z + h*m3)
        l4 = f2(t + h, x + h*k3, y + h*l3, z + h*m3)
        m4 = f3(t + h, x + h*k3, y + h*l3, z + h*m3)

        x, y, z, t = x + h/6*(k1 + 2*k2 + 2*k3 + k4), y + h/6*(l1 + 2*l2 + 2*l3 + l4), z + h/6*(m1 + 2*m2 + 2*m3 + m4), t + h 
       
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ts.append(t)
        
    return ts,xs,ys,zs
