import os
from pathlib import Path
from ngsolve import *
from netgen.geom2d import SplineGeometry

# SetTestoutFile("test.out")
ngsglobals.msg_level = 0

nu = 1e-3
order = 3

comm = mpi_world

geom = SplineGeometry()
geom.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
geom.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")

from_file = False
mesh_size = 0.3
file_name = "tunnel%f.vol.gz" % mesh_size

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

comm.Barrier()
ngmesh.SetGeometry(geom)
mesh = Mesh(ngmesh)
mesh.Curve(3)

V1 = HDiv(mesh, order=order, dirichlet="inlet|cyl|wall", hodivfree=False)
Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)

V = FESpace([V1, Sigma])
(u, sigma), (v, tau) = V.TnT()

n = specialcf.normal(mesh.dim)

dS = dx(element_boundary=True)

a_integrand = -0.5 / nu * InnerProduct(sigma, tau) * dx \
              + div(sigma) * v * dx + div(tau) * u * dx \
              + -(sigma * n) * n * (v * n) * dS \
              + -(tau * n) * n * (u * n) * dS \
              + nu * div(u) * div(v) * dx

a = BilinearForm(V, eliminate_hidden=False, condense=True)
a += a_integrand
a.Assemble()

gfu = GridFunction(V, name="u")
gfu.vec[:] = 0
uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

res = a.mat.CreateColVector()
res.data = a.mat * gfu.vec
print("normtmp", Norm(res))
