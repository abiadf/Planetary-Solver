'''This module implements the gradient ascent/descent technique to solve equations with 2 parameters'''

import os
import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.system('cls')

def multi_gradient_descent(x, y, Z, direction: str = 'ascent', eta = 0.1, n_tries = 1000, initial_guess = 1):
    '''Equation solver using gradient ascent/descent
    INPUT: 
        - initial_guess: first attempt to guess x, equal to x_0
        - eta: learning rate
        - n_tries: number of tries before stopping
        - direction: gradient ascent (+ve), gradient descent (-ve)
    OUTPUT:
        - x_k: x solution for gradient descent 
        - Y_xk: y solution for gradient descent'''
    
    x_derivative = sp.diff(Z, x)
    y_derivative = sp.diff(Z, y)
    x_k = initial_guess
    y_k = initial_guess

    if direction == 'ascent':
        multiplier = 1
    elif direction == 'descent':
        multiplier = -1
    else:
        raise ValueError("Invalid direction. Use 'ascent' or 'descent'")

    for _ in range(n_tries):
        x_gradient = x_derivative.subs(x, x_k)
        y_gradient = y_derivative.subs(y, y_k)

        x_k = x_k + multiplier * eta * x_gradient
        y_k = y_k + multiplier * eta * y_gradient

    Z_k = Z.subs({x: x_k, y: y_k})
    return x_k, y_k, Z_k


def make_3d_plot(Z, x_start, x_end, y_start, y_end):
    x, y = sp.symbols('x y')

    x_values = np.linspace(x_start, x_end, 100)
    y_values = np.linspace(y_start, y_end, 100)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)

    Z_function = sp.lambdify((x, y), Z, 'numpy') # convert Z from symbolic expression to numerically evaluable
    Z_values   = Z_function(x_mesh, y_mesh)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, y_mesh, Z_values, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Plot of Z')
    plt.show()


# ====================================================
x, y = sp.symbols('x y')
Z = -x**2 - 2*x - .5*y**2 - 2*y + 3

x_k, y_k, Z_k = multi_gradient_descent(x, y, Z, 'ascent', eta = 0.1, n_tries = 100, initial_guess = 1)
print(x_k, y_k, Z_k)
make_3d_plot(Z, -100, 100, -100, 100)

