from ngsolve import *

from solvers.krylovspace import *
import netgen.gui

ngsglobals.msg_level = 0

# viscosity
nu = 0.001

# timestepping parameters
tau = 0.001
tend = 10
from netgen.geom2d import SplineGeometry

order = 2

geo = SplineGeometry()
geo.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
mesh = Mesh(geo.GenerateMesh(maxh=0.05))
mesh.Curve(max(order, 1))

mesh.Curve(2)


def spaces_test(precon="bddc"):
	results = {}
	V1 = HDiv(mesh, order=order, dirichlet="wall|inlet|cyl", RT=True)
	V2 = HCurlDiv(mesh, order=order, dirichlet="outlet")
	Q = L2(mesh, order=order)
	X = FESpace([V1, V2, Q])
	(u, sigma, p), (v, tau, q) = X.TnT()

	n = specialcf.normal(mesh.dim)
	h = specialcf.mesh_size

	# see phd thesis Philip Lederer
	a = BilinearForm(X, symmetric=True)
	a += SymbolicBFI(InnerProduct(sigma, tau))
	a += SymbolicBFI(div(sigma) * v + div(tau) * u)
	a += SymbolicBFI(-(sigma * n) * n * (v * n) - (tau * n) * n * (u * n), element_boundary=True)
	a += SymbolicBFI(div(u) * q + div(v) * p)
	preA = Preconditioner(a, 'direct')

	a.Assemble()

	uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
	gfu = GridFunction(X)
	gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

	velocity = CoefficientFunction(gfu.components[0])
	Draw(velocity, mesh, "vel")
	Draw(gfu.components[2], mesh, "p")
	Draw(Norm(velocity), mesh, "|vel|")

	direct_timer = Timer("Direct Solver")
	direct_timer.Start()
	res = gfu.vec.CreateVector()
	res.data = -a.mat * gfu.vec
	# inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
	gfu.vec.data = MinRes(a.mat, res, preA)
	# gfu.vec.data += inv * res

	Redraw()
	direct_timer.Stop()
	input("jljsdlf")

	return results, V1.ndof + V2.ndof + Q.ndof


# spaces_test(V, Q)
# spaces_test(V, Q, precon="multi")
# exit(0)

import pandas as pd

data = pd.DataFrame()
for a in range(1):
	print("#" * 100, a)
	mesh.Refine()

	print(mesh.nv)

	result, ndofs = spaces_test()
	data = data.append({"method": "curldiv", "ndofs": ndofs, "precon": "bddc", "vert": mesh.nv, **result},
	                   ignore_index=True)
	# data.to_csv("bpcg_test_all_nc.csv")
