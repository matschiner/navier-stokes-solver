from math import pi

import numpy
from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.ngstd import Timer
import exponential_integrators.core as ei_core
from exponential_integrators.rk_implicit_ho import RK_impl
import numpy as np

# import netgen.gui

ngsglobals.msg_level = 0
geo = SplineGeometry()
geo.AddRectangle((-1, -1), (1, 1), bcs=("bottom", "right", "top", "left"))

ngsmesh_filename = "test_05"
try:
    mesh = Mesh(ngsmesh_filename + ".vol.gz")
except Exception as e:
    ngsmesh = geo.GenerateMesh(maxh=0.05)
    mesh = Mesh(ngsmesh)
    ngsmesh.Save(ngsmesh_filename)

fes = H1(mesh, order=5, dirichlet="bottom|right|left|top")
print(fes.ndof)

u, v = fes.TnT()  # TnT : Trial and Test function
tau = 0.02


def exact_sol(time):
    return np.exp(-2 * pi ** 2 * time) * sin(pi * x) * sin(pi * y)


b = CoefficientFunction((2 * y * (1 - x * x), -2 * x * (1 - y * y)))
Draw(b, mesh, "wind")

a = BilinearForm(fes, symmetric=False)
a += SymbolicBFI(0.02 * grad(u) * grad(v) + b * grad(u) * v)
# a += grad(u) * grad(v) * dx
a.Assemble()

m = BilinearForm(fes, symmetric=False)
m += u * v * dx
m.Assemble()

mstar = m.mat.CreateMatrix()

mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
invmstar = mstar.Inverse(freedofs=fes.FreeDofs())

f = LinearForm(fes)
f.Assemble()

gfu = GridFunction(fes)
gfu.Set(sin(pi * x) * sin(pi * y))
Draw(gfu, mesh, "u")
t_end = 1  # time that we want to step over within one block-run
t_current = 0  # time counter within one block-run
res = gfu.vec.CreateVector()
k = 0
krylov_dim = 10
krylov_space = [gfu.vec.CreateVector() for j in range(krylov_dim)]

Mm = Matrix(krylov_dim, krylov_dim)
Am = Matrix(krylov_dim, krylov_dim)
Mm_star_inv = Matrix(krylov_dim, krylov_dim)
y_old = Vector(krylov_dim)
y_update = Vector(krylov_dim)
y_old[:] = 0
y_old[0] = 1

rk_method = RK_impl(krylov_dim, krylov_dim * tau)

with TaskManager():
    while t_current < t_end - 0.5 * tau:
        res.data = tau * f.vec - tau * a.mat * gfu.vec
        gfu.vec.data += invmstar * res
        t_current += tau
        print("time =", round(t_current, 4))

    sol_normal = GridFunction(fes)
    sol_normal.vec[:] = 0
    sol_normal.vec.data += gfu.vec
    t_current = 0
    gfu.Set(sin(pi * x) * sin(pi * y))
    while t_current < t_end - 0.5 * tau:

        krylov_space[k % krylov_dim].data = gfu.vec
        res.data = tau * f.vec - tau * a.mat * gfu.vec
        gfu.vec.data += invmstar * res
        t_current += tau
        print("time =", round(t_current, 4))

        k += 1

        if k % krylov_dim == 0:
            krylov_space = ei_core.gram_schmidt(krylov_space)
            Am, Mm = ei_core.reduced_space_projection_update(krylov_space, a, m, Am, Mm)

            y_update = rk_method.do_step_ngs(-Mm, Am, y_old)

            gfu.vec[:] = 0
            for i in range(krylov_dim):
                gfu.vec.data += y_update[i] * krylov_space[i]

            Redraw(blocking=True)

    sol_normal.vec.data -= gfu.vec
    print(Norm(sol_normal.vec), )
