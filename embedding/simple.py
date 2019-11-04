from ngsolve import *
from ngsolve.internal import visoptions
from ngsolve.la import EigenValues_Preconditioner
from netgen.geom2d import SplineGeometry, unit_square

import netgen.gui


def tang(v):
    return v - (v * n) * n


mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

nu = 1
order = 5
condense = True
alpha = 10
force = CoefficientFunction((x, x))

V = HDiv(mesh, order=order, dirichlet="right|bottom|left|top")
Vhat = TangentialFacetFESpace(mesh, order=order, dirichlet="right|bottom|left|top")

n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

dS = dx(element_boundary=True)

X = FESpace([V, Vhat])
(u, uhat), (v, vhat) = X.TnT()
v_dual = v.Operator("dual")

VH1 = VectorH1(mesh, order=order, dirichlet="right|bottom|left|top")
vH1trial, vH1test = VH1.TnT()

M = BilinearForm(trialspace=VH1, testspace=X)
M += vH1trial * v_dual * dx
M += vH1trial * v_dual * dS
M += vH1trial * tang(vhat) * dS
M.Assemble()

Mw = BilinearForm(X, X)
Mw += u * v_dual * dx
Mw += u * v_dual * dS
Mw += uhat * tang(vhat) * dS
Mw.Assemble()

Mw_inverse = Mw.mat.Inverse(inverse="umfpack", freedofs=X.FreeDofs(condense))

Mw_trans = Mw.mat.CreateTranspose()
Mw_trans_inverse = Mw_trans.Inverse(inverse="umfpack", freedofs=X.FreeDofs(condense))

proj = Projector(X.FreeDofs(condense), True, )

E = Mw_inverse @ M.mat
ET = M.mat.T @ Mw_trans_inverse

# , freedofs=BitArray([True for k in range(V.ndof)] + [False for k in range(Vhat.ndof)]))
# tmp = M.mat.CreateColVector()
# tmp.data = M.mat * generic_H1_func.vec

# blocks = [list(X.GetDofNrs(f)) for f in mesh.facets] \
#         + [list(X.GetDofNrs(m)) for m in mesh.Elements()]

# blockjac = Mw.mat.CreateBlockSmoother(blocks)

# blockjac.Smooth(generic_H1_func_transformed.vec, tmp)
# generic_H1_func_transformed.vec.data = Mw_inverse* tmp

laplaceH1 = BilinearForm(VH1, condense=condense)
laplaceH1 += InnerProduct(nu * grad(vH1trial), grad(vH1test)) * dx
laplaceH1.Assemble()

laplaceH1_inverse = laplaceH1.mat.Inverse(freedofs=VH1.FreeDofs(condense))

Ahat_inv = E @ laplaceH1_inverse @ ET

gradu = CoefficientFunction((grad(u),), dims=(2, 2))
gradv = CoefficientFunction((grad(v),), dims=(2, 2))

a = BilinearForm(X, condense=condense)
a += SymbolicBFI(nu * InnerProduct(gradu, gradv))
a += SymbolicBFI(nu * InnerProduct(gradu.trans * n, tang(vhat - v)), element_boundary=True)
a += SymbolicBFI(nu * InnerProduct(gradv.trans * n, tang(uhat - u)), element_boundary=True)
a += SymbolicBFI(nu * alpha * (order + 1) ** 2 / h * InnerProduct(tang(vhat - v), tang(uhat - u)), element_boundary=True)

a.Assemble()
xfree = X.FreeDofs(condense)
pre_block = a.mat.CreateBlockSmoother([
    [d for d in dofnrs if xfree[d]] for (e, dofnrs) in zip(mesh.Elements(), [X.GetDofNrs(e) for e in mesh.Elements()]) if len(dofnrs) > 0
])
pre_jacobi = a.mat.CreateSmoother(X.FreeDofs(condense))
# preA = c.mat
preA = Ahat_inv + pre_block
preA = pre_block + Ahat_inv

f = LinearForm(X)
f += SymbolicLFI(force * v)
f.Assemble()

gfu = GridFunction(X)

iterative = True
if iterative:
    if a.condense:
        f.vec.data += a.harmonic_extension_trans * f.vec
        solvers.CG(mat=a.mat, pre=preA, rhs=f.vec, sol=gfu.vec, maxsteps=40)
        gfu.vec.data += a.inner_solve * f.vec
        gfu.vec.data += a.harmonic_extension * gfu.vec
    else:
        solvers.CG(mat=a.mat, pre=preA, rhs=f.vec, sol=gfu.vec, maxsteps=40)
else:
    gfu.vec.data = a.mat.Inverse(X.FreeDofs(True), inverse="sparsecholesky") * f.vec

vel = gfu.components[0]

Draw(vel, mesh, "vel")
Draw(div(vel), mesh, "div")
visoptions.scalfunction = 'vel:0'
input("end")
