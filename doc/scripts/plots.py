import os

import numpy as np
import matplotlib.pyplot as plt

DOC_DIRECTORY = os.path.dirname(os.path.abspath(f"{__file__}/.."))
FIGURES_DIRECTORY = f"{DOC_DIRECTORY}/figures"


def gradient_descent_basic():
    learning_rate = 0.009
    x_coordinates = [-5.]
    for i in range(5):
        x = x_coordinates[-1]
        x_coordinates.append(x-learning_rate*4*x**3)
    return x_coordinates


def plot_gradient_descent():
    t = np.arange(-5., 5., 0.2)

    x_coordinates = gradient_descent_basic()
    y_coordinates = list(map(lambda x: x**4, x_coordinates))
    print(x_coordinates)
    print(y_coordinates)
    plt.plot(x_coordinates, y_coordinates)

    plt.plot(t, t ** 4)
    plt.savefig(f"{FIGURES_DIRECTORY}/gradient_descent.png")


plot_gradient_descent()
