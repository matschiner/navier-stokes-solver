import numpy as np
# import matplotlib.pyplot as plt
# import scipy.special as sp
import scipy.integrate as integrate
from ngsolve import *
from scipy.linalg import block_diag


def lagrange(c, n, x):
    return np.prod(c[:n] - x, axis=0) * np.prod(c[n + 1:] - x, axis=0) / np.prod(c[:n] - c[n]) / np.prod(c[n + 1:] - c[n])


class RK_impl():

    def __init__(self, deg=3, h=0.1):
        self.deg = deg
        self.h = h
        c_big_interval, self.b_numpy = np.polynomial.legendre.leggauss(deg)
        self.c = (c_big_interval + 1) / 2
        self.b_numpy /= deg
        ca = np.array([self.c]).T

        self.a = np.zeros((deg, deg))
        self.b = np.zeros(deg)
        for i in range(deg):
            self.b[i] = integrate.quad(lambda xx: lagrange(ca, i, xx), 0, 1)[0]
            for j in range(deg):
                self.a[i, j] = integrate.quad(lambda xx: lagrange(ca, j, xx), 0, self.c[i])[0]

    def do_step(self, M, A, y_old):
        lhs1 = block_diag(*([M] * self.deg))
        lhs2 = np.block([[aij * A for aij in self.a[i, :]] for i in range(self.deg)])

        m_star = lhs1 - self.h * lhs2
        rhs = np.block(([np.dot(A, y_old)] * self.deg))
        k = np.dot(np.linalg.inv(m_star), rhs)
        kk = k.reshape((self.deg, len(y_old))).T  # using row indexing per default
        return y_old + self.h * np.dot(kk, self.b)

    def do_step_ngs(self, Mm, Am, ym):
        res = Vector(len(ym))
        nv = self.do_step(np.array(Mm), np.array(Am), np.array(ym))
        for i in range(len(nv)):
            res[i] = nv[i]
        return res


"""def legendre(n, x):
    x = x * 2 - 1
    return (2 ** n) * sum([(x ** k) * sp.binom(n, k) * sp.binom((n + k - 1) / 2, n) for k in range(n + 1)])
"""

if __name__ == "__main__":
    h = 0.01
    rk = RK_impl(15, h)
    # print(rk)
    # print("resulting a rounded\n", np.round(rk.a, 4))
    # print("resulting a\n", rk.a)
    # print("integration points c", rk.c)
    # print("integration weights b numpy", rk.b_numpy)
    # print("integration weights b", rk.b)

    M = np.zeros((2, 2))
    M[:] = 0
    M[0, 0] = 1
    M[1, 1] = 1

    A = np.zeros((2, 2))
    A[:] = 0
    A[0, 0] = -1
    A[1, 1] = -2

    y = np.zeros(2)
    y[:] = 1

    # T = Matrix(2, 2)
    # T[:] = 0
    # T[0, 1] = 1
    # print("T\n", T, "--")
    # rk.do_step_ngs(T, A, y)
    # exit(0)
    for i in range(10):
        print("%.3f" % (i * h), np.round(y, 4))
        y = rk.do_step(M, A, y)

    print("%.3f" % ((i + 1) * h), np.round(y, 4))
