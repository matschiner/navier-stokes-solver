from ngsolve import *
from ngsolve.internal import visoptions
from ngsolve.la import EigenValues_Preconditioner
from netgen.geom2d import SplineGeometry, unit_square

import netgen.gui

from embedding.helpers import CreateEmbeddingPreconditioner


def tang(v):
    return v - (v * n) * n


mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

nu = 1
order = 5
alpha = 10
force = CoefficientFunction((x, x))

diri = "right|bottom|left|top"
V = HDiv(mesh, order=order, dirichlet=diri)
Vhat = TangentialFacetFESpace(mesh, order=order, dirichlet=diri)

n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

dS = dx(element_boundary=True)

X = FESpace([V, Vhat])
(u, uhat), (v, vhat) = X.TnT()
v_dual = v.Operator("dual")

Ahat_inv = CreateEmbeddingPreconditioner(X, nu, diri=diri)

grad_u = CoefficientFunction((grad(u),), dims=(2, 2))
grad_v = CoefficientFunction((grad(v),), dims=(2, 2))

a_integrand = nu * InnerProduct(grad_u, grad_v) * dx \
              + nu * InnerProduct(grad_u.trans * n, tang(vhat - v)) * dS \
              + nu * InnerProduct(grad_v.trans * n, tang(uhat - u)) * dS \
              + nu * alpha * (order + 1) ** 2 / h * InnerProduct(tang(vhat - v), tang(uhat - u)) * dS

a = BilinearForm(X, condense=True)
a += a_integrand
a.Assemble()

a_full = BilinearForm(X, condense=False)
a_full += a_integrand
a_full.Assemble()

xfree = X.FreeDofs(True)
pre_block = a.mat.CreateBlockSmoother([
    [d for d in dofnrs if xfree[d]] for (e, dofnrs) in zip(mesh.Elements(), [X.GetDofNrs(e) for e in mesh.Elements()]) if len(dofnrs) > 0
])
pre_jacobi = a.mat.CreateSmoother(X.FreeDofs(True))
preA = Ahat_inv + pre_block

f = LinearForm(X)
f += SymbolicLFI(force * v)
f.Assemble()

gfu = GridFunction(X)

I = IdentityMatrix()
a_extended = (I + a.harmonic_extension) @ preA @ (I + a.harmonic_extension_trans) + a.inner_solve
solvers.CG(mat=a_full.mat, rhs=f.vec, pre=a_extended, sol=gfu.vec, maxsteps=40)
vel = gfu.components[0]

Draw(vel, mesh, "vel")
Draw(div(vel), mesh, "div")
visoptions.scalfunction = 'vel:0'
input("end")

"""
class HarmonicExtension(BaseMatrix):
    def __init__(self):
        super(HarmonicExtension, self).__init__()
        pass

    def MultAdd(self, c, x, y):
        tmp = preA.CreateColVector()
        y.data += a.inner_solve * x
        x.data += a.harmonic_extension_trans * x
        tmp.data = preA * x
        y.data += tmp
        y.data += a.harmonic_extension * tmp

    def CreateRowVector(self):
        return a.harmonic_extension.CreateRowVector()

    def CreateColVector(self):
        return a.harmonic_extension.CreateColVector()


a_extended = HarmonicExtension()
"""
