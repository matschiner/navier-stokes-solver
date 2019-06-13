from ngsolve import *

from solvers.krylovspace import *
from solvers.krylovspace import MinRes
from solvers.bramblepasciak import BramblePasciakCG as BPCG
from multiplicative_precond.preconditioners import MultiplicativePrecond

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

mesh = Mesh(geo.GenerateMesh(maxh=0.1))

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
    p, q = Q.TnT()

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
    #a += +1e-8 * p * q * dx

    b = BilinearForm(trialspace=V, testspace=Q)
    b += SymbolicBFI(div(u) * q)

    bramblePasciakTimer = Timer("BramblePasciakCG")
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

    else:
        preA = Preconditioner(a, 'local')
        a.Assemble()
        b.Assemble()

    preconTimer.Stop()
    results["preconTime"] = preconTimer.time

    mp = BilinearForm(Q)
    mp += SymbolicBFI(p * q)
    preM = Preconditioner(mp, 'local')
    mp.Assemble()

    f = LinearForm(V)
    f += SymbolicLFI(CoefficientFunction((0, x - 0.5)) * v)
    f.Assemble()

    g = LinearForm(Q)
    g.Assemble()

    gfu = GridFunction(V, name="u")
    gfp = GridFunction(Q, name="p")
    uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    gfu.components[2].Set(uin, definedon=mesh.Boundaries("inlet"))
    sol = BlockVector([gfu.vec, gfp.vec])

    sol2 = sol.CreateVector()
    sol2[0].data = gfu.vec
    sol2[1].data = gfp.vec

    # Draw(x - 0.5, mesh, "source")

    with TaskManager():  # pajetrace=100 * 1000 * 1000):
        bramblePasciakTimer.Start()
        results["nits_bpcg"] = BPCG(a.mat, b.mat, None, f.vec, g.vec, preA, preM, sol2, initialize=False, tol=1e-6, maxsteps=100000, rel_err=True)
        bramblePasciakTimer.Stop()
        results["time_bpcg"] = bramblePasciakTimer.time

    K = BlockMatrix([[a.mat, b.mat.T], [b.mat, None]])
    C = BlockMatrix([[preA, None], [None, preM]])
    rhs = BlockVector([f.vec, g.vec])
    sol = BlockVector([gfu.vec, gfp.vec])

    with TaskManager():  # pajetrace=100*1000*1000):
        minResTimer.Start()
        tmp, results["nits_minres"] = MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False, tol=1e-6, maxsteps=100000)
        minResTimer.Stop()
        results["time_minres"] = minResTimer.time

    print("BramblePasciakCGMax took", round(bramblePasciakTimer.time, 4), "seconds")
    print("MinRes took", round(minResTimer.time, 4), "seconds")

    sol2.data -= sol
    print("difference", Norm(sol2))
    return results, V.ndof + Q.ndof


# spaces_test(V, Q)
# spaces_test(V, Q, precon="multi")
# exit(0)

import pandas as pd

data = pd.DataFrame()
for j in range(4):
    print("#" * 100, j)
    mesh.Refine()

    print(mesh.nv)

    result, ndofs = spaces_test()
    data = data.append({"method": "curldiv", "ndofs": ndofs, "precon": "bddc", "vert": mesh.nv, **result}, ignore_index=True)
    data.to_csv("bpcg_test_all_nc.csv")