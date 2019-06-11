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
tau = 0.2


def exact_sol(time):
    return np.exp(-2 * pi ** 2 * time) * sin(pi * x) * sin(pi * y)


solutions = {}
err_exact = {}
err_exact_taus = {}
for sol_index in range(7):
    tau /= 2
    time = 0.0
    print("tau", tau, "\n")
    b = CoefficientFunction((2 * y * (1 - x * x), -2 * x * (1 - y * y)))
    Draw(b, mesh, "wind")

    a = BilinearForm(fes, symmetric=False)
    # a += SymbolicBFI(0.02 * grad(u) * grad(v) + b * grad(u) * v)
    a += grad(u) * grad(v) * dx
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
    # gfu.Set(sin(pi * x) * sin(pi * y))
    gfu.Set((1 - y * y) * x)
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

    tau_big = krylov_dim * tau

    timer_ol = Timer("OuterLoop")
    timer_ol.Start()

    rk_method = RK_impl(krylov_dim, tau * krylov_dim)
    method = "exp-int"

    with TaskManager():
        while t_current < t_end - 0.5 * tau:

            krylov_space[k % krylov_dim].data = gfu.vec
            res.data = tau * f.vec - tau * a.mat * gfu.vec
            gfu.vec.data += invmstar * res
            t_current += tau
            print("\rtime =", round(time + t_current, 4), end="")

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
                else:
                    y_update = rk_method.do_step_ngs(-Mm, Am, y_old)

                # updating the big system with the better solution
                gfu.vec[:] = 0
                for i in range(krylov_dim):
                    gfu.vec.data += y_update[i] * krylov_space[i]

                sol = np.array(gfu.vec[:])
                solutions[sol_index] = sol
                # timer_ol.Start()
                Redraw(blocking=True)
                # input("next step")
        err_exact[sol_index] = sqrt(Integrate((exact_sol(t_current) - gfu) ** 2, mesh))
        err_exact_taus[sol_index] = tau
        # print("diff", err_exact[sol_index])

    # input("next")
    time += t_current

err = np.ones(sol_index - 1)
taus = np.ones(sol_index - 1)
for s in range(sol_index - 1):
    e = np.linalg.norm(solutions[s] - sol)
    err[s] = e
    taus[s] = tau * (2 ** (sol_index - s - 1))
    print(tau * (2 ** (sol_index - s - 1)), "vs", tau, e)
import matplotlib.pyplot as plt

plt.loglog(taus, err)
exact_taus = np.array([err_exact_taus[k] for k in err_exact_taus])
# plt.loglog(exact_taus, np.array([err_exact[k] for k in err_exact]))
plt.loglog(exact_taus, 1e4 * exact_taus ** 4)
plt.loglog(exact_taus, 1e12 * exact_taus ** krylov_dim)
plt.legend(["ref err", "tau^4", f"tau^{krylov_dim}"])
plt.show()
