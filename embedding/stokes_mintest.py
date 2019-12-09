import os
from pathlib import Path
from ngsolve import *
import netgen.gui
from netgen.geom2d import SplineGeometry
from solvers.krylovspace import *

SetTestoutFile("test.out")
ngsglobals.msg_level = 0

# viscosity
nu = 1e-3

order = 2

comm = mpi_world
rank = comm.rank
np = comm.size



precon = "embedded"
geom_name = "tunnel"
slip = True
inflow = None
slip_boundary = ["cyl", "wall"]

geom = SplineGeometry()
geom.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
geom.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
# ngmesh = geom.GenerateMesh(maxh=0.036)
diri = "inlet" + ("" if slip else "|" + ("|".join(slip_boundary)))
inflow = "inlet"

from_file = False
mesh_size = 0.3
file_name = "tunnel%f.vol.gz" % mesh_size
if rank == 0:

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

if rank != 0:
    if comm.Sum(0) == 0:
        ngmesh = netgen.meshing.Mesh.Receive(comm)
    else:
        ngmesh = netgen.meshing.Mesh(dim=2, comm=comm)
        ngmesh.Load(file_name)

comm.Barrier()
ngmesh.SetGeometry(geom)
mesh = Mesh(ngmesh)
# print(mesh.nv)
# print("ne",mesh.ne, mesh.nedge, mesh.nv)
# quit()
mesh.Curve(3)

condense = True
V1 = HDiv(mesh, order=order, dirichlet=diri + "|" + ("|".join(slip_boundary)), hodivfree=False)
Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)
VHat = TangentialFacetFESpace(mesh, order=order - 1, dirichlet="inlet|outlet" + ("" if slip else "|".join(slip_boundary)))
Q = L2(mesh, order=order - 1)
Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
#Sigma = Compress(Sigma)

if mesh.dim == 2:
    S = L2(mesh, order=order - 1)
else:
    S = VectorL2(mesh, order=order - 1)

S.SetCouplingType(IntRange(0, S.ndof), COUPLING_TYPE.HIDDEN_DOF)
#S = Compress(S)

V = FESpace([V1, VHat, Sigma, S])

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

a_full = BilinearForm(V, eliminate_hidden=False, condense=True, elmatev=True, print=True, printelmat=True)
a_full += a_integrand
a_full.Assemble()

f = LinearForm(V)
f.Assemble()

gfu = GridFunction(V, name="u")
gfu.vec[:] = 0
if geom_name == "tunnel":
    uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    gfu.components[0].Set(uin, definedon=mesh.Boundaries(inflow))

res = a_full.mat.CreateColVector()
res.data = a_full.mat * gfu.vec
print("normtmp", Norm(res))
