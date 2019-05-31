import numpy as np
import scipy.integrate as integrate
from ngsolve import *


def lagrange(c, n, x):
    return np.prod(c[:n] - x, axis=0) * np.prod(c[n + 1:] - x, axis=0) / np.prod(c[:n] - c[n]) / np.prod(c[n + 1:] - c[n])


class ImplicitRungeKuttaMethodWeights():
    def __init__(self, deg=3):
        c_big_interval, self.b_numpy = np.polynomial.legendre.leggauss(deg)
        self.c = (c_big_interval + 1) / 2
        self.b_numpy /= deg
        ca = np.array([self.c]).T

        self.a = Matrix(deg, deg)
        self.b = Vector(deg)
        for i in range(deg):
            self.b[i] = integrate.quad(lambda xx: lagrange(ca, i, xx), 0, 1)[0]
            for j in range(deg):
                self.a[i, j] = integrate.quad(
                    lambda xx: lagrange(ca, j, xx), 0, self.c[i])[0]


def linear_implicit_runge_kutta_step(weights, matrix, current_value, step_width):
    temp = matrix * current_value

    coefficient_matrix = Matrix(matrix.Width() * weights.a.Height(),
                                matrix.Height() * weights.a.Height())
    for block_col in range(weights.a.Width()):
        for block_row in range(weights.a.Height()):
            a = weights.a[block_row, block_col]
            for inner_col in range(matrix.Width()):
                for inner_row in range(matrix.Height()):
                    row = block_row * matrix.Height() + inner_row
                    col = block_col * matrix.Width() + inner_col
                    coefficient_matrix[row, col] = - \
                        step_width * a * matrix[inner_row, inner_col]

    for i in range(matrix.Height() * weights.a.Width()):
        coefficient_matrix[i, i] += 1

    inverse = Matrix(coefficient_matrix.Width(), coefficient_matrix.Height())
    coefficient_matrix.Inverse(inverse)

    inhomogeneity = Vector(matrix.Height() * weights.a.Height())
    for i in range(weights.a.Height()):
        inhomogeneity[i * matrix.Width():(i + 1) * matrix.Width()] = temp

    k = inverse * inhomogeneity

    next_value = Vector(len(current_value))
    next_value[:] = current_value
    for i in range(weights.a.Height()):
        next_value += step_width * weights.b[i] * \
            k[i * matrix.Width():(i + 1) * matrix.Width()]

    return next_value
