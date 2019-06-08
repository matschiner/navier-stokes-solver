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

tau = 1e-3

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
u.Set(exp(- (20 ** 2) * ((x - 0.5) ** 2 + (y - 0.5) ** 2)))
Draw(u, autoscale=False, min=-0.1, max=0.1)

a.Assemble()
m_star.Assemble()
m.Assemble()
f.Assemble()

ms_inv = m_star.mat.Inverse(fes.FreeDofs())

v = GridFunction(fes)
v.vec[:] = 0

r = u.vec.CreateVector()
accel = u.vec.CreateVector()

k = 0
krylov_dim = 16
krylov_space = [u.vec.CreateVector() for j in range(krylov_dim + 1)]
# ... consisting of all u vectors plus first v vector in last position

Mm = Matrix(krylov_dim + 1, krylov_dim + 1)
Am = Matrix(krylov_dim + 1, krylov_dim + 1)
Mm_star_inv = Matrix(krylov_dim + 1, krylov_dim + 1)

rk_method = RK_impl(krylov_dim, krylov_dim * tau)

v_0_small = Vector(krylov_dim + 1)
v_0_small[:] = 0
v_0_small[0] = 1

u_0_small = Vector(krylov_dim + 1)
u_0_small[:] = 0
u_0_small[krylov_dim] = 1
with TaskManager():
    while t < t_end:
        krylov_space[k % krylov_dim].data = u.vec
        if k % krylov_dim == 0:
            krylov_space[krylov_dim] = v.vec
            pass
        r.data = -tau * a.mat * u.vec - 0.5 * tau ** 2 * a.mat * v.vec
        accel.data = ms_inv * r
        u.vec.data += tau * v.vec + 0.5 * tau * accel
        v.vec.data += accel
        t += tau
        print("t =", round(t, 4))
        if t == tau:
            continue

        k += 1
        if k % krylov_dim == 0:
            krylov_space = ei_core.gram_schmidt(krylov_space)
            Am, Mm = ei_core.reduced_space_projection_update(krylov_space, a, m, Am, Mm)

            v_update = rk_method.do_step_ngs(Mm, -Am, v_0_small)
            u_update = rk_method.do_step_ngs(Mm, np.identity(krylov_dim + 1), u_0_small)

            # updating the big system with the better solution
            v.vec[:] = 0
            u.vec[:] = 0
            for i in range(krylov_dim + 1):
                v.vec.data += v_update[i] * krylov_space[i]
                u.vec.data += u_update[i] * krylov_space[i]

                # input("next step")
        Redraw(blocking=True)
        #input("ljsldf")