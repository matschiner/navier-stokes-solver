import netgen.gui
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve.ngstd import Timer
from exponential_integrators.rk_implicit_ho import RK_impl
import numpy as np
import exponential_integrators.core as ei_core

ngsmesh_filename = "wave_04"
try:
    mesh = Mesh(ngsmesh_filename + ".vol.gz")
except Exception as e:
    ngsmesh = unit_square.GenerateMesh(maxh=0.04)
    mesh = Mesh(ngsmesh)
    ngsmesh.Save(ngsmesh_filename)

Draw(mesh)

tau = 0.01

t = 0
t_end = 100

fes = H1(mesh, order=5, dirichlet="left|right|top|bottom")
u, phi = fes.TnT()

a = BilinearForm(fes)
a += SymbolicBFI(grad(u) * grad(phi))

m_star = BilinearForm(fes)
m_star += SymbolicBFI(u * phi + 0.25 * tau ** 2 * grad(u) * grad(phi))

m = BilinearForm(fes)
m += SymbolicBFI(u * phi)

f = LinearForm(fes)

u = GridFunction(fes)
u.Set(exp(- (20 ** 2) * ((x - 0.5) ** 2 + (y - 0.5) ** 2)))
Draw(u)

a.Assemble()
m_star.Assemble()
m.Assemble()
f.Assemble()

ms_inv = m_star.mat.Inverse(fes.FreeDofs())

r = u.vec.CreateVector()
v = u.vec.CreateVector()
v[:] = 0
accel = u.vec.CreateVector()

k = 0
krylov_dim = 6
krylov_space = [u.vec.CreateVector() for j in range(krylov_dim)]

Mm = Matrix(krylov_dim, krylov_dim)
Am = Matrix(krylov_dim, krylov_dim)
Mm_star_inv = Matrix(krylov_dim, krylov_dim)
y_old = Vector(krylov_dim)
y_update = Vector(krylov_dim)
y_old[:] = 0
y_old[0] = 1

tau_big = krylov_dim * tau

rk_method = RK_impl(krylov_dim, tau)
method = "rk"
v_0_small = Vector(krylov_dim)
v_0_small[:] = 0

while t < t_end:
    krylov_space[k % krylov_dim].data = u.vec
    t += tau
    print("t =", round(t, 4))
    r.data = -tau * a.mat * u.vec - 0.5 * tau ** 2 * a.mat * v
    accel.data = ms_inv * r
    u.vec.data += tau * v + 0.5 * tau * accel
    v.data += accel

    k += 1
    if k % krylov_dim == 0:

        krylov_space = ei_core.gram_schmidt(krylov_space)
        Am, Mm = ei_core.reduced_space_projection_update(krylov_space, a, m, Am, Mm)

        y_update = v_0_small
        for i in range(krylov_dim):
            y_update = rk_method.do_step_ngs(-Mm, Am, y_update)
        v_0_small = y_update
        # print("update2\n", y_update)

        # updating the big system with the better solution
        v[:] = 0
        for i in range(krylov_dim):
            v.data += y_update[i] * krylov_space[i]

            # input("next step")
        Redraw(blocking=True)
