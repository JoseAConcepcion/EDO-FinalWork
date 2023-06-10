from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def plotting(t: list, x: list, y: list, z: list, tilte="") -> None:
    plt.title(tilte)
    plt.plot(t, x, label='presas jovenes', color='green')
    plt.plot(t, y, label='presas jovenes', color='brown')
    plt.plot(t, z, label='depredadores', color='red')
    plt.legend()
    plt.show()
    return


def animation(t: list, y1: list, y2: list, y3: list, title="") -> None:

    fig = plt.figure()
    line1, = plt.plot(t, y1, label='presas jovenes', color="green")
    line2, = plt.plot(t, y2, label='presas adultas', color="brown")
    line3, = plt.plot(t, y3, label='depredadores', color="red")
    plt.legend()

    def update(frame):
        y1[:-1] = y1[1:]
        y2[:-1] = y2[1:]
        y3[:-1] = y3[1:]

        line1.set_ydata(y1)
        line2.set_ydata(y2)
        line3.set_ydata(y3)
        return line1, line2, line3

    plt.title(title)
    ani = FuncAnimation(fig, update, frames=len(t), interval=75, repeat=False)
    plt.show()
    return

def dynamic_behaviour(x: list, y: list, z: list) -> None:
    
    X = np.array(x)
    Y = np.array(y)
    Z = np.array(z)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('presas jovenes')
    ax.set_ylabel('presas adultas')
    ax.set_zlabel('depredadores')
    ax.plot(X, Y, Z) 
    plt.show()

    return
