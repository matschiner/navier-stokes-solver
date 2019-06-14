from ngsolve import *

from solvers.krylovspace import *
from solvers.krylovspace import MinRes
from solvers.bramblepasciak import BramblePasciakCG as BPCG
from multiplicative_precond.preconditioners import MultiplicativePrecond
import netgen.gui
ngsglobals.msg_level = 0

# viscosity
nu = 0.001

# timestepping parameters
tau = 0.001
tend = 10
from netgen.geom2d import SplineGeometry

geo = SplineGeometry()
geo.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")

mesh = Mesh(geo.GenerateMesh(maxh=0.05))

mesh.Curve(3)


def spaces_test(precon="bddc"):
	results = {}

	order = 2

	V1 = HDiv(mesh, order=order, dirichlet="wall|inlet|cyl")
	Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)
	VHat = VectorFacet(mesh, order=order - 1, dirichlet="wall|inlet|cyl")
	Q = L2(mesh, order=order - 1)
	Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
	Sigma = Compress(Sigma)
	V = FESpace([V1, Sigma, VHat])

	(u, sigma, u_hat), (v, tau, v_hat) = V.TnT()

	n = specialcf.normal(mesh.dim)

	def tang(vec):
		return vec - (vec * n) * n

	a = BilinearForm(V, eliminate_hidden=True)
	a += -InnerProduct(sigma, tau) * dx
	a += div(sigma) * v * dx + div(tau) * u * dx
	a += -(sigma * n) * n * (v * n) * dx(element_boundary=True)
	a += -(tau * n) * n * (u * n) * dx(element_boundary=True)
	a += -(tau * n) * tang(u_hat) * dx(element_boundary=True)
	a += -(sigma * n) * tang(v_hat) * dx(element_boundary=True)
	#a += div(u) * q * dx + div(v) * p * dx
	#a += 1e-12 * p * q * dx
	a += 1e0 * div(u) * div(v) * dx

	preA = Preconditioner(a, 'bddc')
	a.Assemble()

	# mp = BilinearForm(Q)
	# mp += SymbolicBFI(p * q)
	# preM = Preconditioner(mp, 'local')
	# mp.Assemble()

	f = LinearForm(V)
	f += SymbolicLFI(CoefficientFunction((0, x - 0.5)) * v)
	f.Assemble()

	gfu = GridFunction(V, name="u")
	uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
	gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

	# Draw(x - 0.5, mesh, "source")

	with TaskManager():  # pajetrace=100 * 1000 * 1000):
		results["nits_bpcg"] = CG(a.mat, f.vec, preA, gfu.vec, initialize=False)
	Draw(gfu.components[0], mesh, "v")
	#Draw(gfu.components[3], mesh, "p")
	return results, V.ndof + Q.ndof


result, ndofs = spaces_test()
input("end")
