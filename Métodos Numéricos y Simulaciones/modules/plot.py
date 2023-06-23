from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import colorsys as c

def plotting(t: list, x: list, y: list, z: list, tilte="") -> None:
    plt.figure()
    plt.title(tilte)
    plt.plot(t, x, label='presas jóvenes', color='green')
    plt.plot(t, y, label='presas adultas', color='brown')
    plt.plot(t, z, label='depredadores', color='red')
    plt.legend()
    plt.show()
    return


def animation(t: list, y1: list, y2: list, y3: list, title="") -> None:
    fig = plt.figure()
    line1, = plt.plot(t, y1, label='presas jóvenes', color="green")
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
    ani = FuncAnimation(fig, update, frames=len(t), interval=5, repeat=False)
    plt.show()
    return


def create_fig3d(xlabel: str, ylabel: str, zlabel: str) -> plt:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return ax


def plot3d_All(datas, title="") -> None:
    ax = create_fig3d("Presas Jóvenes", "Presas Adultas", "Depredadores")
    color = ['red', 'blue', 'black', 'green', 'yellow']
    for i in range(len(datas)):
        solution = datas[i]
        plot3d(ax, solution[:, 0], solution[:, 1], solution[:, 2], color[i])
    plt.show()
    return


def dynamic_behaviour(x: list, y: list, z: list) -> None:
    ax = create_fig3d("Presas Jóvenes", "Presas Adultas", "Depredadores")
    plot3d(ax, x, y, z)
    plt.show()
    return


def plot3d(ax: plt, x: list, y: list, z: list, color="blue") -> None:
    ax.plot(x, y, z, color=color)
    return