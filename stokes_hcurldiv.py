import netgen.gui
from ngsolve import *
from netgen.geom2d import SplineGeometry
#from minres import MinRes
from ngsolve.solvers import MinRes
from bramble_pasciak_cg import bramble_pasciak_cg

order = 2

geo = SplineGeometry()
geo.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
mesh = Mesh(geo.GenerateMesh(maxh=0.06))
mesh.Curve(max(order, 1))
Draw(mesh)


V1 = HDiv(mesh, order=order, dirichlet="wall|inlet|cyl")
Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)
VHat = TangentialFacetFESpace(mesh, order=order - 1, dirichlet="wall|inlet|cyl")
Q = L2(mesh, order=order - 1)
Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
Sigma = Compress(Sigma)
X = FESpace([V1, VHat, Sigma])
(u, u_hat, sigma), (v, v_hat, tau) = X.TnT()

n = specialcf.normal(mesh.dim)


def tang(vec):
    return vec - (vec * n) * n


a = BilinearForm(X, eliminate_hidden=True, condense=True)
a += -InnerProduct(sigma, tau) * dx
a += div(sigma) * v * dx + div(tau) * u * dx
a += -(sigma * n) * n * (v * n) * dx(element_boundary=True)
a += -(tau * n) * n * (u * n) * dx(element_boundary=True)
a += -(tau * n) * tang(u_hat) * dx(element_boundary=True)
a += -(sigma * n) * tang(v_hat) * dx(element_boundary=True)
a += div(u) * div(v) * dx

p, q = Q.TnT()

b = BilinearForm(trialspace=X, testspace=Q)
b += div(u) * q * dx

preA = Preconditioner(a, 'bddc')

m = BilinearForm(Q)
m += p * q * dx

preM = Preconditioner(m, 'local')


a.Assemble()
b.Assemble()
m.Assemble()

f = LinearForm(X)
f += SymbolicLFI(CoefficientFunction((0, x - 0.5)) * v)
f.Assemble()

g = LinearForm(Q)
g.Assemble()

grid_function = GridFunction(X)
grid_function_pressure = GridFunction(Q)
uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
grid_function.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

Draw(grid_function.components[0])
Draw(grid_function_pressure)

solution = BlockVector([grid_function.vec, grid_function_pressure.vec])
bramble_pasciak_cg(a.mat, b.mat, None, preA, preM,
                   f.vec, g.vec, solution=solution, max_steps=10000)

Redraw()
input("press enter to stop")
