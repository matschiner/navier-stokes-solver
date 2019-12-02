import sys

from ngsolve import *
from ngsolve.la import EigenValues_Preconditioner
from solvers.krylovspace import *
from solvers.krylovspace import MinRes
from solvers.bramblepasciak import BramblePasciakCG as BPCG
from multiplicative_precond.preconditioners import MultiplicativePrecond
import netgen.gui
from embedding.helpers import CreateEmbeddingPreconditioner
from blockjacobi_parallel import *
from ngsolve.meshes import MakeStructured2DMesh

ngsglobals.msg_level = 0

# viscosity
nu = 1e-3

order = 3

comm = mpi_world
rank = comm.rank
np = comm.size

from netgen.geom2d import SplineGeometry

precon = "embedded"
geom_name = "tunnel"
slip = True
inflow = None
if geom_name == "tunnel":
    geom = SplineGeometry()
    geom.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
    geom.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
    # ngmesh = geom.GenerateMesh(maxh=0.036)
    diri = "inlet" + ("" if slip else "|wall|cyl")
    inflow = "inlet"
elif geom_name == "stretched":
    geom = None
    diri = "left|top|bottom"
    inflow = "left"
else:
    geom = netgen.geom2d.unit_square
    diri = "top|bottom"
    inflow = "left"

if rank == 0:
    if geom:
        ngmesh = geom.GenerateMesh(maxh=0.036)
        if comm.size > 1:
            ngmesh.Distribute(comm)
        mesh = Mesh(ngmesh)
    else:
        mesh = MakeStructured2DMesh(nx=4 * 3, ny=10 * 3, secondorder=True, quads=False, mapping=lambda x, y: (5 * x, y))
else:
    ngmesh = netgen.meshing.Mesh.Receive(comm)
    ngmesh.SetGeometry(geom)
    mesh = Mesh(ngmesh)

mesh.Curve(5)
condense = True

V1 = HDiv(mesh, order=order, dirichlet=diri+"|cyl|wall", hodivfree=False)
Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)
VHat = TangentialFacetFESpace(mesh, order=order - 1, dirichlet="inlet|outlet")
Q = L2(mesh, order=order - 1)
Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
Sigma = Compress(Sigma)

if mesh.dim == 2:
    S = L2(mesh, order=order - 1)
else:
    S = VectorL2(mesh, order=order - 1)

S.SetCouplingType(IntRange(0, S.ndof), COUPLING_TYPE.HIDDEN_DOF)
S = Compress(S)

if precon == "bddc":
    for e in mesh.edges:
        dofs = V1.GetDofNrs(e)
        V1.SetCouplingType(dofs[1], COUPLING_TYPE.WIREBASKET_DOF)

V = FESpace([V1, VHat, Sigma, S])

if precon == "bddc":
    for e in mesh.edges:
        dofs = V1.GetDofNrs(e)
        V.SetCouplingType(dofs[1], COUPLING_TYPE.WIREBASKET_DOF)

(u, u_hat, sigma, W), (v, v_hat, tau, R) = V.TnT()
p, q = Q.TnT()

n = specialcf.normal(mesh.dim)


def tang(vec):
    return vec - (vec * n) * n


if mesh.dim == 2:
    def Skew2Vec(m):
        return m[1, 0] - m[0, 1]
else:
    def Skew2Vec(m):
        return CoefficientFunction((m[0, 1] - m[1, 0], m[2, 0] - m[0, 2], m[1, 2] - m[2, 1]))

dS = dx(element_boundary=True)

a_integrand = -0.5 / nu * InnerProduct(sigma, tau) * dx \
              + (InnerProduct(W, Skew2Vec(tau)) + InnerProduct(R, Skew2Vec(sigma))) * dx \
              + div(sigma) * v * dx + div(tau) * u * dx \
              + -(sigma * n) * n * (v * n) * dS \
              + -(tau * n) * n * (u * n) * dS \
              + -(tau * n) * tang(u_hat) * dS \
              + -(sigma * n) * tang(v_hat) * dS \
              + nu * div(u) * div(v) * dx #\
              #+ 10 ** 10 * u.Trace() * n * v.Trace() * n * ds("cyl|wall")  # \
# + 10 ** 9 * u_hat.Trace() * v_hat.Trace() * ds("cyl|wall")

a = BilinearForm(V, eliminate_hidden=True, condense=True)
a += a_integrand
a_full = BilinearForm(V, eliminate_hidden=True, condense=False)
a_full += a_integrand

b = BilinearForm(trialspace=V, testspace=Q)
b += div(u) * q * dx

minResTimer = Timer("MyMinRes")
preconTimer = Timer("Precon")

preconTimer.Start()

if precon == "embedded":
    Ahat_inv = CreateEmbeddingPreconditioner(V, nu, diri=diri)

    a.Assemble()
    b.Assemble()

    x_free = V.FreeDofs(condense)

    blocks = [[d for d in dofnrs if x_free[d]] for dofnrs in (V.GetDofNrs(e) for e in mesh.facets) if len(dofnrs) > 0]  # \
    # + [list(d for d in ar if d >= 0 and x_free[d]) for ar in (V.GetDofNrs(NodeId(FACE, k)) for k in range(mesh.nface))]

    if comm.size > 1:
        precon_blockjac = BlockJacobiParallel(a.mat.local_mat, blocks)
    else:
        precon_blockjac = a.mat.CreateBlockSmoother(blocks)
    preA = precon_blockjac + Ahat_inv
else:
    preA = Preconditioner(a, 'bddc')
    a.Assemble()
    b.Assemble()

a_full.Assemble()

I = IdentityMatrix()
a_extended = (I + a.harmonic_extension) @ preA @ (I + a.harmonic_extension_trans) + a.inner_solve

evals = list(EigenValues_Preconditioner(a.mat, preA))
print(evals)
print(evals[0], evals[-1], "cond", evals[-1] / evals[0])

preconTimer.Stop()

mp = BilinearForm(Q)
mp += 0.5 / nu * p * q * dx
preM = Preconditioner(mp, 'local')
mp.Assemble()

f = LinearForm(V)
# f += SymbolicLFI(CoefficientFunction((0, x - 0.5)) * v)
f.Assemble()

g = LinearForm(Q)
g.Assemble()

#import numpy as np
#
#gftest = GridFunction(V, name="ev")
#gftest.vec.FV().NumPy()[:] = np.random.rand(len(gftest.vec))
#
#print(gftest.vec)
#Draw(gftest.components[0], mesh, "ev")
#
#prod = Ahat_inv @ a_full.mat
#
#for i in range(10):
#    gftest.vec.data = prod * gftest.vec
#
#Redraw()
#input("ljsldf")

gfu = GridFunction(V, name="u")
gfp = GridFunction(Q, name="p")
if geom_name == "tunnel":
    uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    gfu.components[0].Set(uin, definedon=mesh.Boundaries(inflow))
else:
    uin = CoefficientFunction((1.5 * 4 * y * (1 - y) / (0.41 * 0.41), 0))
    gfu.components[0].Set(uin, definedon=mesh.Boundaries(inflow))

K = BlockMatrix([[a_full.mat, b.mat.T], [b.mat, None]])
C = BlockMatrix([[a_extended, None], [None, preM]])
rhs = BlockVector([f.vec, g.vec])
sol = BlockVector([gfu.vec, gfp.vec])

with TaskManager():  # pajetrace=100*1000*1000):
    minResTimer.Start()
    MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False, tol=1e-9, maxsteps=1000)
    minResTimer.Stop()

timers = dict((t["name"], t["time"]) for t in Timers())
print("MinRes", timers["MyMinRes"])
print("Precon", timers["Precon"])

Draw(gfu.components[0], mesh, "v")
Draw(gfp, mesh, "p")
input("finish")
# Draw(gfu.components[1], mesh, "v_hat")
