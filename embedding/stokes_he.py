import os
import sys
from pathlib import Path

from ngsolve import *
from ngsolve.la import EigenValues_Preconditioner
from solvers.krylovspace import *
from solvers.krylovspace import MinRes

from embedding.helpers import CreateEmbeddingPreconditioner
from ngsolve.meshes import MakeStructured2DMesh

ngsglobals.msg_level = 0

if mpi_world.size == 1:
    import netgen.gui
# viscosity
nu = 1e-3

order = 3

comm = mpi_world
rank = comm.rank
np = comm.size

from netgen.geom2d import SplineGeometry

num_refinements = int(sys.argv[1])
precon = "embedded"
auxiliary_precon = "direct"
geom_name = "tunnel"
slip = True 
inflow = None

slip_boundary = ["wall"]
if geom_name == "tunnel":
    geom = SplineGeometry()
    geom.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
    geom.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
    # ngmesh = geom.GenerateMesh(maxh=0.036)
    diri = "inlet|cyl" + ("" if slip else "|" + ("|".join(slip_boundary)))
    inflow = "inlet"
elif geom_name == "stretched":
    geom = None
    diri = "left|top|bottom"
    inflow = "left"
else:
    geom = netgen.geom2d.unit_square
    diri = "top|bottom"
    inflow = "left"

from_file = False
mesh_size = 0.05
file_name = "tunnel%f.vol.gz" % mesh_size
if rank == 0:
    if geom:

        file = Path(file_name)

        if os.path.isfile(file_name):
            print("loaded mesh")
            from_file = True
            ngmesh = netgen.meshing.Mesh(dim=2, comm=comm)
            comm.Sum(1)
            ngmesh.Load(file_name)
        else:
            print("generating mesh")
            ngmesh = geom.GenerateMesh(maxh=mesh_size)
            ngmesh.Save(file_name)
            comm.Sum(0)
        if comm.size > 1 and not from_file:
            ngmesh.Distribute(comm)

    else:
        mesh = MakeStructured2DMesh(nx=4 * 3, ny=10 * 3, secondorder=True, quads=False, mapping=lambda x, y: (5 * x, y))
if rank != 0:
    if comm.Sum(0) == 0:
        ngmesh = netgen.meshing.Mesh.Receive(comm)
    else:
        ngmesh = netgen.meshing.Mesh(dim=2, comm=comm)
        ngmesh.Load(file_name)

comm.Barrier()
ngmesh.SetGeometry(geom)
for n in range(num_refinements):
    ngmesh.Refine()
mesh = Mesh(ngmesh)
mesh.Curve(3)

condense = True
V1 = HDiv(mesh, order=order, dirichlet=diri + "|" + ("|".join(slip_boundary)), hodivfree=False)
Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)
VHat = TangentialFacetFESpace(mesh, order=order - 1, dirichlet="inlet|outlet" if slip else ".*")
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
              + nu * div(u) * div(v) * dx  # \
# + 10 ** 10 * u.Trace() * n * v.Trace() * n * ds("cyl|wall")  # \
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
    Ahat_inv = CreateEmbeddingPreconditioner(V, nu, diri=diri, slip=slip, slip_boundary=slip_boundary, auxiliary_precon=auxiliary_precon)

    a.Assemble()
    b.Assemble()

    x_free = V.FreeDofs(condense)

    blocks = [[d for d in dofnrs if x_free[d]] for dofnrs in (V.GetDofNrs(e) for e in mesh.facets) if len(dofnrs) > 0]  # \
    # + [list(d for d in ar if d >= 0 and x_free[d]) for ar in (V.GetDofNrs(NodeId(FACE, k)) for k in range(mesh.nface))]

    if comm.size > 1:
        precon_blockjac_unwrapped = a.mat.local_mat.CreateBlockSmoother(blocks, parallel=True)
        precon_blockjac = ParallelMatrix(
            precon_blockjac_unwrapped,
            row_pardofs=a.mat.row_pardofs,
            col_pardofs=a.mat.col_pardofs,
            op=ParallelMatrix.D2D,
        )
    else:
        precon_blockjac = a.mat.CreateBlockSmoother(blocks)

    preA = Ahat_inv + precon_blockjac
else:
    preA = Preconditioner(a, 'bddc')
    a.Assemble()
    b.Assemble()

a_full.Assemble()

I = IdentityMatrix()
if comm.size > 1:
    op_tmp1 = ParallelMatrix(
        I + a.harmonic_extension,
        row_pardofs=a.mat.row_pardofs,
        col_pardofs=a.mat.col_pardofs,
        op=ParallelMatrix.C2C,
    )
    op_tmp2 = ParallelMatrix(
        I + a.harmonic_extension_trans,
        row_pardofs=a.mat.row_pardofs,
        col_pardofs=a.mat.col_pardofs,
        op=ParallelMatrix.D2D,
    )
    op_tmp3 = ParallelMatrix(
        a.inner_solve,
        row_pardofs=a.mat.row_pardofs,
        col_pardofs=a.mat.col_pardofs,
        op=ParallelMatrix.D2C,
    )
    a_extended = op_tmp1 @ preA @ op_tmp2 + op_tmp3
else:
    a_extended = (I + a.harmonic_extension) @ preA @ (I + a.harmonic_extension_trans) + a.inner_solve

preconTimer.Stop()

evals = list(EigenValues_Preconditioner(a.mat, preA))
# print(evals)
cond = evals[-1] / evals[0]
if comm.rank==0:
     print(evals[0], evals[-1], "cond", cond)

mp = BilinearForm(Q)
mp += 0.5 / nu * p * q * dx
preM = Preconditioner(mp, 'local')
mp.Assemble()

f = LinearForm(V)
# f += SymbolicLFI(CoefficientFunction((0, x - 0.5)) * v)
f.Assemble()

g = LinearForm(Q)
g.Assemble()

gfu = GridFunction(V, name="u")
gfp = GridFunction(Q, name="p")
gfu.vec[:] = 0
gfp.vec[:] = 0
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
    _, nits = MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False, tol=1e-9, maxsteps=10000, printrates=comm.rank == 0)
    minResTimer.Stop()

timers = dict((t["name"], t["time"]) for t in Timers())

result_stats = {
    "timeMinRes": timers["MyMinRes"],
    "timePrecon": timers["Precon"],
    "nits": nits,
    "numElements": mesh.ne,
    "numDofs": V.ndof,
    "cond": cond
}

if comm.rank == 1 or comm.size == 1:
    import pprint

    pprint.pprint(result_stats)

print("norm", Norm(gfu.vec))
if comm.size == 1:
    Draw(gfu.components[0], mesh, "v")
    Draw(gfp, mesh, "p")
    input("finish")
# Draw(gfu.components[1], mesh, "v_hat")


# vtk = VTKOutput(ma=mesh, coefs=[gfu.components[0][0],gfu.components[0][1], gfp], names=["solu0","solu1","solp"], filename="vtkout/vtkout_p"+str(rank), subdivision=2)
# vtk.Do()
