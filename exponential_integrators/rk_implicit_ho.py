import numpy as np
# import matplotlib.pyplot as plt
# import scipy.special as sp
import scipy.integrate as integrate
from ngsolve import *


def lagrange(c, n, x):
    return np.prod(c[:n] - x, axis=0) * np.prod(c[n + 1:] - x, axis=0) / np.prod(c[:n] - c[n]) / np.prod(c[n + 1:] - c[n])


class RK_impl():

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
                self.a[i, j] = integrate.quad(lambda xx: lagrange(ca, j, xx), 0, self.c[i])[0]


"""def legendre(n, x):
    x = x * 2 - 1
    return (2 ** n) * sum([(x ** k) * sp.binom(n, k) * sp.binom((n + k - 1) / 2, n) for k in range(n + 1)])
"""

if __name__ == "main":
    rk = RK_impl(3)
    print(rk)
    print("resulting a rounded\n", np.round(rk.a, 4))
    print("resulting a\n", rk.a)
    print("integration points c", rk.c)
    print("integration weights b numpy", rk.b_numpy)
    print("integration weights b", rk.b)

    """for i in range(deg):
        plt.plot(xx, lagrange(ca, i, xx))
    plt.show()"""
    print("jlsjdf")
