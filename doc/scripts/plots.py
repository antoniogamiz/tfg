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


def sigmoid_plot():
    plt.clf()
    x = np.linspace(-5, 5, 101)
    def sigmoid(x): return 1/(1+np.exp(-x))
    plt.plot(x, sigmoid(x), label=r'$\sigma(x)$')
    plt.plot(x, sigmoid(x)*sigmoid(-x), 'r--', label=r"$\sigma'(x)$")
    plt.legend()
    plt.savefig(f"{FIGURES_DIRECTORY}/sigmoid.png")


def hiperbolic_tangent_plot():
    plt.clf()
    x = np.linspace(-5, 5, 101)

    plt.plot(x, np.tanh(x), label=r'$f(x)$')
    plt.plot(x, (1 / np.cosh(x)) ** 2, 'r--', label=r"$f'(x)$")
    plt.legend()
    plt.savefig(f"{FIGURES_DIRECTORY}/tanh.png")


def relu_plot():
    plt.clf()
    x = np.linspace(-2, 2, 101)
    def relu(x): return np.maximum(x, 0)
    def relu_der(x): return np.maximum((-x)/x, 0)

    plt.plot(x, relu(x), label=r'$f(x)$')
    plt.plot(x, relu_der(x), 'r--', label=r"$f'(x)$")
    plt.legend()
    plt.savefig(f"{FIGURES_DIRECTORY}/relu.png")


# plot_gradient_descent()
sigmoid_plot()
hiperbolic_tangent_plot()
relu_plot()
