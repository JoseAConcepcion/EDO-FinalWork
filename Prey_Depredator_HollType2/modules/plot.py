from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


def plot_simulations(title: str, datas: list) -> None:
    
    n = len(datas)
    fig, axes = plt.subplots(3, n, figsize=(15, 15))
    fig.suptitle(title)

    for i in range(n):
        t, v = datas[i] 
      
        x = [v[j][0] for j in range(len(v))]
        y = [v[j][1] for j in range(len(v))]
        z = [v[j][2] for j in range(len(v))]

        plotting(axes[i], t, x, y, z)
        break
    fig.show()
    return


def plotting(ax: plt, t: list, x: list, y: list, z: list) -> None:
    ax.plot(t, x, label='presas jovenes', color='green')
    ax.plot(t, y, label='presas jovenes', color='brown')
    ax.plot(t, z, label='depredadores', color='red')
    ax.legend()
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
