# not currently working ... testing in wave_simple

import netgen.gui
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve.ngstd import Timer
from exponential_integrators.rk_implicit_ho import RK_impl
import numpy as np
import exponential_integrators.core as ei_core

ngsmesh_filename = "wave_05"
try:
    mesh = Mesh(ngsmesh_filename + ".vol.gz")
except Exception as e:
    ngsmesh = unit_square.GenerateMesh(maxh=0.05)
    mesh = Mesh(ngsmesh)
    ngsmesh.Save(ngsmesh_filename)

Draw(mesh)
solutions = {}
solutions_index = 0
tau = 0.2
for i in range(8):
    tau /= 2
    print(tau, type(tau))
    t = 0
    t_end = 1

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
    u.Set(exp(- (20 ** 2) * ((x - 0.5) ** 2 + (y - 0.5) ** 2)))  # + exp(- (20 ** 2) * ((x - 0.6) ** 2 + (y - 0.7) ** 2)))
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

    timer_ol = Timer("OuterLoop")
    timer_ol.Start()

    rk_method = RK_impl(krylov_dim, tau_big)
    method = "rk"

    while t < t_end:
        krylov_space[k % krylov_dim].data = u.vec
        t += tau
        print("t =", t)
        r.data = -tau * a.mat * u.vec - 0.5 * tau ** 2 * a.mat * v
        accel.data = ms_inv * r
        u.vec.data = u.vec + tau * v + 0.5 * tau * accel
        v.data = v + accel
        k += 1
        if k % krylov_dim == 0:
            # timer_ol.Stop()
            # print("building base took", timer_ol.time)
            # timer_ol = Timer("OuterLoop")
            krylov_space = ei_core.gram_schmidt(krylov_space)

            Am, Mm = ei_core.reduced_space_projection_update(krylov_space, a, m, Am, Mm)

            if method == "impl_EV":
                Mm_star = Mm + tau_big * Am
                Mm_star.Inverse(Mm_star_inv)
                y_update = -tau_big * Mm_star_inv * Am * y_old
                y_update[0] += 1
                # print("update\n", y_update)
            else:
                y_update = y_old
                for i in range(krylov_dim):
                    y_update = rk_method.do_step_ngs(-Mm, Am, y_update)
                # print("update2\n", y_update)

            # updating the big system with the better solution
            u.vec[:] = 0
            for i in range(krylov_dim):
                u.vec.data += y_update[i] * krylov_space[i]

            # timer_ol.Start()
            sol = np.array(u.vec[:])
            solutions[solutions_index] = sol

            # np.savez("sol-wave%.3f" % np.log10(tau), sol)

            # input("next step")
        Redraw(blocking=True)
    solutions_index += 1
    input("next")

err = np.ones(solutions_index - 1)
taus = np.ones(solutions_index - 1)
for s in range(solutions_index - 1):
    e = np.linalg.norm(solutions[s] - sol)
    err[s] = e
    taus[s] = tau * (2 ** (solutions_index - s - 1))
    print(tau * (2 ** (solutions_index - s - 1)), "vs", tau, e)
import matplotlib.pyplot as plt

plt.loglog(taus, err)
plt.loglog(taus, taus * taus)
plt.loglog(taus, taus ** krylov_dim)
plt.show()
input("\nend")
