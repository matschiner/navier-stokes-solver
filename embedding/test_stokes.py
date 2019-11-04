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

ngsglobals.msg_level = 0

# viscosity
nu = 1  # 1e-3

# timestepping parameters
tau = 0.001
tend = 10
order = 3

comm = mpi_world
rank = comm.rank
np = comm.size

from netgen.geom2d import SplineGeometry

geom = SplineGeometry()
geom.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
geom.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
diri = "wall|inlet|cyl"

geom = netgen.geom2d.unit_square
diri = "left"

if rank == 0:
    # master proc generates and immediately distributes
    ngmesh = geom.GenerateMesh(maxh=0.1)

    if comm.size > 1:
        ngmesh.Distribute(comm)
else:
    # worker procs receive mesh
    ngmesh = netgen.meshing.Mesh.Receive(comm)
    # and then manually set the geometry
    ngmesh.SetGeometry(geom)
mesh = Mesh(ngmesh)

# mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
mesh.Curve(3)
condense = True


def spaces_test(precon="bddc"):
    results = {}

    V1 = HDiv(mesh, order=order, dirichlet=diri)
    Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)
    VHat = TangentialFacetFESpace(mesh, order=order - 1, dirichlet=".*")
    Q = L2(mesh, order=order - 1)
    Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
    Sigma = Compress(Sigma)

    V = FESpace([V1, VHat, Sigma])

    (u, u_hat, sigma), (v, v_hat, tau) = V.TnT()
    p, q = Q.TnT()

    n = specialcf.normal(mesh.dim)

    def tang(vec):
        return vec - (vec * n) * n

    dS = dx(element_boundary=True)

    a = BilinearForm(V, eliminate_hidden=True, condense=condense)
    a += -1 / nu * InnerProduct(sigma, tau) * dx
    a += div(sigma) * v * dx + div(tau) * u * dx
    a += -(sigma * n) * n * (v * n) * dS
    a += -(tau * n) * n * (u * n) * dS
    a += -(tau * n) * tang(u_hat) * dS
    a += -(sigma * n) * tang(v_hat) * dS
    a += nu * div(u) * div(v) * dx

    b = BilinearForm(trialspace=V, testspace=Q)
    b += div(u) * q * dx

    minResTimer = Timer("MinRes")
    preconTimer = Timer("Precon")

    preconTimer.Start()
    if precon == "multi":
        print("multiplicative Precond")
        a.Assemble()
        b.Assemble()

        preJpoint = a.mat.CreateSmoother(V.FreeDofs())

        vertexdofs = BitArray(V.ndof)
        vertexdofs[:] = False

        for vert in mesh.vertices:
            for dofs_nr in V.GetDofNrs(vert):
                vertexdofs[dofs_nr] = True

        vertexdofs &= V.FreeDofs()

        preCoarse = a.mat.Inverse(vertexdofs, inverse="sparsecholesky")
        preA = MultiplicativePrecond(preJpoint, preCoarse, a.mat)
    elif precon == "embedded":
        Ahat_inv = CreateEmbeddingPreconditioner(V, nu, diri=diri)

        a.Assemble()
        b.Assemble()

        t1 = a.mat.CreateRowVector()
        t2 = a.mat.CreateRowVector()
        t2.data = Ahat_inv * t1

        # pre_jacobi = a.mat.CreateSmoother(X.FreeDofs(condense))

        x_free = V.FreeDofs(condense)
        # blocks = [[d for d in dofnrs if d
        # > 0 and x_free[d]] for (e, dofnrs) in zip(mesh.Elements(), [V.GetDofNrs(e) for e in mesh.Elements()])]
        # blocks = [[d for d in dofnrs if x_free[d]] for dofnrs in (V.GetDofNrs(e) for e in mesh.facets) if len(dofnrs) > 0]

        blocks = [[d for d in dofnrs if x_free[d]] for dofnrs in (V.GetDofNrs(e) for e in mesh.facets) if len(dofnrs) > 0]  # \
        # + [list(d for d in ar if d >= 0 and x_free[d]) for ar in (V.GetDofNrs(NodeId(FACE, k)) for k in range(mesh.nface))]

        # blocks = [list(d for d in ar if d >= 0 and x_free[d]) for ar in (V.GetDofNrs(e) for e in mesh.Elements())]

        if mpi_world.size > 1:
            precon = BlockJacobiParallel(a.mat.local_mat, blocks)
            preA = precon + Ahat_inv

        else:
            pre_blockjacobi = a.mat.CreateBlockSmoother(blocks) if mpi_world.size == 1 else a.mat.local_mat.CreateBlockSmoother(blocks, parallel=True)
            preA = pre_blockjacobi + Ahat_inv
    else:
        preA = Preconditioner(a, 'bddc')
        a.Assemble()
        b.Assemble()

    evals = list(EigenValues_Preconditioner(a.mat, preA))
    print(evals[0], evals[-1], "cond", evals[-1] / evals[0])

    preconTimer.Stop()
    results["preconTime"] = preconTimer.time

    mp = BilinearForm(Q)
    mp += 1 / nu * p * q * dx
    preM = Preconditioner(mp, 'local')
    mp.Assemble()

    f = LinearForm(V)
    # f += SymbolicLFI(CoefficientFunction((0, x - 0.5)) * v)
    f.Assemble()

    g = LinearForm(Q)
    g.Assemble()

    gfu = GridFunction(V, name="u")
    gfp = GridFunction(Q, name="p")
    uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

    # Draw(x - 0.5, mesh, "source")
    if a.condense:
        f.vec.data += a.harmonic_extension_trans * f.vec

    K = BlockMatrix([[a.mat, b.mat.T], [b.mat, None]])
    C = BlockMatrix([[preA, None], [None, preM]])
    rhs = BlockVector([f.vec, g.vec])
    sol = BlockVector([gfu.vec, gfp.vec])

    #tmp1 = a.mat.CreateColVector()
    #tmp1[:] = 1
    #CG(mat=a.mat, pre=preA, rhs=tmp1, sol=gfu.vec, printrates=True)

    with TaskManager():  # pajetrace=100*1000*1000):
        minResTimer.Start()
        tmp, results["nits_minres"] = MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False, tol=1e-9, maxsteps=1000)
        if a.condense:
            sol[0].data += a.harmonic_extension * sol[0]
            sol[0].data += a.inner_solve * f.vec

        minResTimer.Stop()
        results["time_minres"] = minResTimer.time

    print("MinRes took", round(minResTimer.time, 4), "seconds")

    # gfu.vec.data = sol[0]
    # gfp.vec.data = sol[1]
    Draw(gfu.components[0], mesh, "v")
    Draw(gfp, mesh, "p")
    input("test")
    # Draw(gfu.components[1], mesh, "v_hat")

    return results, V.ndof + Q.ndof


spaces_test("bddc")
# spaces_test(V, Q, precon="multi")

# import pandas as pd

# data = pd.DataFrame()
# for j in range(1):
#     print("#" * 100, j)
#     # mesh.Refine()

#     print(mesh.nv)

#     result, ndofs = spaces_test()
# # data = data.append({"method": "curldiv", "ndofs": ndofs, "precon": "bddc", "vert": mesh.nv, **result}, ignore_index=True)
# # data.to_csv("bpcg_test_all_nc.csv")
