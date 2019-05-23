from math import pi
from ngsolve import *
from netgen.geom2d import SplineGeometry

import netgen.gui

geo = SplineGeometry()
geo.AddRectangle((-1, -1), (1, 1), bcs=("bottom", "right", "top", "left"))
mesh = Mesh(geo.GenerateMesh(maxh=0.1))
Draw(mesh)
fes = H1(mesh, order=3, dirichlet="bottom|right|left|top")

u, v = fes.TnT()  # TnT : Trial and Test function

time = 0.0
tau = 0.001
CRANK_NICOLSON = "cn15125"
IMPL_EULER = "implev215"
method = IMPL_EULER
method_scaling_factor = +.5 if method == CRANK_NICOLSON else 1

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
mstar.AsVector().data = m.mat.AsVector() + tau * method_scaling_factor * a.mat.AsVector()
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
krylov_dim = 5
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
M_tmp = m.mat.CreateColVector()
A_tmp = a.mat.CreateColVector()
tau_big = krylov_dim * tau
with TaskManager():
	while t_intermediate < tstep - 0.5 * tau:

		krylov_space[k % krylov_dim].data = gfu.vec
		res.data = tau * f.vec - tau * a.mat * gfu.vec
		gfu.vec.data += invmstar * res
		t_intermediate += tau
		print(time + t_intermediate)
		Redraw(blocking=True)
		k += 1

		if k % krylov_dim == 0:
			for i in range(krylov_dim):
				for j in range(krylov_dim):
					M_tmp.data = m.mat * krylov_space[j]
					A_tmp.data = a.mat * krylov_space[j]
					Mm[i, j] = InnerProduct(M_tmp, krylov_space[i])
					Am[i, j] = InnerProduct(A_tmp, krylov_space[i])
			Mm_star = Mm + tau_big * Am
			Mm_star.Inverse(Mm_star_inv)
			y_update = -tau_big * Mm_star_inv * y_old
			gfu.vec[:] = 0
			for i in range(krylov_dim):
				gfu.vec.data += (y_update[i] + (1 if i == 0 else 0)) * krylov_space[i]

print("", Norm(gfu.vec))
time += t_intermediate
