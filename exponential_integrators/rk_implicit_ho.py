import numpy as np

deg = 3
c_big_interval, b = np.polynomial.legendre.leggauss(deg)
c = (c_big_interval + 1) / 2
b /= deg

ips = 100000
x = np.linspace(0, 1, ips + 1)

legendre_coeffs = np.zeros(deg)

comp = np.array([
    [0.25, 0.25 - np.sqrt(3) / 6],
    [0.25 + np.sqrt(3) / 6, 0.25],
])
"""for i in range(deg):
    for j in range(deg):
        legendre_coeffs[:] = 0
        legendre_coeffs[i] = 1
        y = np.polynomial.legendre.legval(x[:-1], legendre_coeffs)
        integral = 0.5 / ips * sum(y[:-1] + y[1:])
        poly=np.polynomial.legendre.Legendre(np.ones(10))
        print("poly",poly.basis(deg))
    print("i", sum(y) / ips, integral)"""

import scipy.special as sp
import scipy.integrate as integrate


def legendre(n, x):
    x = x * 2 - 1
    return (2 ** n) * sum([(x ** k) * sp.binom(n, k) * sp.binom((n + k - 1) / 2, n) for k in range(n + 1)])


a = np.zeros((deg, deg))
for i in range(deg):
    for j in range(deg):
        a[i, j] = integrate.quad(lambda x: legendre(j, x), 0, c[i])[0]
print("resulting a\n", np.round(a,4))
print("integration points c", c)
print("integration weights b", b)
print("compared", comp)
pass

print()
