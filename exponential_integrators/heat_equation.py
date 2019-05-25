from math import pi

import numpy
from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.ngstd import Timer

import netgen.gui

ngsglobals.msg_level = 3
geo = SplineGeometry()
geo.AddRectangle((-1, -1), (1, 1), bcs=("bottom", "right", "top", "left"))

ngsmesh_filename = "test_1"
try:
    mesh = Mesh(ngsmesh_filename + ".vol.gz")
except Exception as e:
    ngsmesh = geo.GenerateMesh(maxh=0.1)
    mesh = Mesh(ngsmesh)
    ngsmesh.Save(ngsmesh_filename)

fes = H1(mesh, order=3, dirichlet="bottom|right|left|top")
print(fes.ndof)

u, v = fes.TnT()  # TnT : Trial and Test function

time = 0.0
tau = 0.001

b = CoefficientFunction((2 * y * (1 - x * x), -2 * x * (1 - y * y)))
Draw(b, mesh, "wind")

a = BilinearForm(fes, symmetric=False)
a += SymbolicBFI(0.01 * grad(u) * grad(v) + b * grad(u) * v)
a.Assemble()

m = BilinearForm(fes, symmetric=False)
m += SymbolicBFI(u * v)
m.Assemble()

mstar = m.mat.CreateMatrix()

mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()
invmstar = mstar.Inverse(freedofs=fes.FreeDofs())

f = LinearForm(fes)
gaussp = exp(-6 * ((x + 0.5) * (x + 0.5) + y * y)) - exp(-6 * ((x - 0.5) * (x - 0.5) + y * y))
Draw(gaussp, mesh, "f")
f += SymbolicLFI(gaussp * v)
f.Assemble()

gfu = GridFunction(fes)
gfu.Set((1 - y * y) * x)
Draw(gfu, mesh, "u")

tstep = 1  # time that we want to step over within one block-run
t_intermediate = 0  # time counter within one block-run
res = gfu.vec.CreateVector()
t_bigstep = t_intermediate
k = 0
krylov_dim = 100
krylov_space = {}
for j in range(krylov_dim):
    krylov_space[j] = gfu.vec.CreateVector()

Mm = Matrix(krylov_dim, krylov_dim)
Am = Matrix(krylov_dim, krylov_dim)
Mm_star_inv = Matrix(krylov_dim, krylov_dim)
y = Vector(krylov_dim)
y_old = Vector(krylov_dim)
y_update = Vector(krylov_dim)
y_old[:] = 0
y_old[0] = 1

tau_big = krylov_dim * tau
timer_ol = Timer("OuterLoop")
timer_ol.Start()


def gram_schmidt(space, tries=3):
    for tries in range(tries):
        for j in range(len(space)):
            for i in range(0, j):
                space[j].data -= InnerProduct(space[i], space[j]) / InnerProduct(space[i], space[i]) * space[i]
            # normalising
            # krylov_space[j].data = 1 / Norm(krylov_space[j]) * krylov_space[j]

    # checking the orthogonality
    # Orthogonality = Matrix(len(krylov_space), len(krylov_space))
    # for i in range(len(krylov_space)):
    #    for j in range(len(krylov_space)):
    #        Orthogonality[i, j] = InnerProduct(krylov_space[i], krylov_space[j])
    # print("orthogonality\n", numpy.round(Orthogonality,12))


def reduced_space_projection_update(space, Am, Mm):
    # timer_prep = Timer("InnerProducts")
    # timer_prep.Start()

    M_tmp = m.mat.CreateColVector()
    A_tmp = a.mat.CreateColVector()
    for j in range(len(space)):
        M_tmp.data = m.mat * krylov_space[j]
        A_tmp.data = a.mat * krylov_space[j]
        for i in range(len(space)):
            Mm[i, j] = InnerProduct(M_tmp, krylov_space[i])
            Am[i, j] = InnerProduct(A_tmp, krylov_space[i])
    # timer_prep.Stop()
    # print("time of inner products", timer_prep.time)
    return Am, Mm


with TaskManager():
    while t_intermediate < tstep - 0.5 * tau:

        krylov_space[k % krylov_dim].data = gfu.vec
        res.data = tau * f.vec - tau * a.mat * gfu.vec
        gfu.vec.data += invmstar * res
        t_intermediate += tau
        print("time =", round(time + t_intermediate, 4))

        k += 1

        if k % krylov_dim == 0:
            timer_ol.Stop()
            print("building base took", timer_ol.time)
            krylov_space = gram_schmidt(krylov_space)

            reduced_space_projection_update(krylov_space, Am, Mm)

            # solving the reduced system
            Mm_star = Mm + tau_big * Am
            Mm_star.Inverse(Mm_star_inv)
            y_update = -tau_big * Mm_star_inv * Am * y_old

            # updating the big system with the better solution
            gfu.vec[:] = 0
            for i in range(krylov_dim):
                gfu.vec.data += (y_update[i] + (1 if i == 0 else 0)) * krylov_space[i]

            timer_ol.Start()
            Redraw(blocking=True)

print("", Norm(gfu.vec))
time += t_intermediate
