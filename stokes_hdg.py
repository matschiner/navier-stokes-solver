from ngsolve.la import *
from solvers.krylovspace import *
from solvers.bramblepasciak import BramblePasciakCG as BPCG
from ngsolve import *

ngsglobals.msg_level = 0

# viscosity
nu = 0.001

# timestepping parameters
tau = 0.001
tend = 10
from multiplicative_precond.preconditioners import MultiplicativePrecond
from netgen.geom2d import SplineGeometry

geo = SplineGeometry()
geo.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")

mesh = Mesh(geo.GenerateMesh(maxh=0.05))

mesh.Curve(3)


class elements:
    def __init__(self, mesh):
        self.mesh = mesh

    def bdm(self, order=2):
        self.order = order
        self.v1, self.v2, self.setup_lambda = (
            FESpace([HDiv(self.mesh, order=order),
                     VectorFacet(self.mesh, order=order, dirichlet="wall|inlet|cyl"),
                     ]),
            L2(self.mesh, order=0),
            lambda v1, v2: 1,
        )
        return self

    def setup(self):
        self.setup_lambda(self.v1, self.v2)
        return self.v1, self.v2


element_names = [a for a in filter(lambda x: "__" not in x and x not in ["setup", "mesh"], dir(elements(mesh)))]


def spaces_test(V, Q, order, precon="bddc",
                block=True):
    results = {}
    alpha = 10

    results = {}

    (u, uhat), (v, vhat) = V.TnT()
    p, q = Q.TnT()

    gradu = CoefficientFunction((grad(u),), dims=(2, 2)).trans
    gradv = CoefficientFunction((grad(v),), dims=(2, 2)).trans

    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size

    def tang(vec):
        return vec - (vec * n) * n

    a = BilinearForm(V, symmetric=False)
    a += SymbolicBFI(InnerProduct(gradu, gradv))
    a += SymbolicBFI(10000 * u * v)
    a += SymbolicBFI(InnerProduct(gradu * n, tang(vhat - v)), element_boundary=True)
    a += SymbolicBFI(InnerProduct(gradv * n, tang(uhat - u)), element_boundary=True)
    a += SymbolicBFI(alpha * order * order / h * InnerProduct(tang(vhat - v), tang(uhat - u)), element_boundary=True)

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

        vertexdofs = BitArray(V.ndof)
        vertexdofs[:] = False

        if (block):
            # Block Jacobi
            blocks = []
            freedofs = V.FreeDofs(False)
            for vert in mesh.vertices:
                vdofs = set()
                for edge in mesh[vert].edges:
                    vdofs |= set(dof for dof in V.GetDofNrs(edge) if freedofs[dof])
                for el in mesh[vert].faces:
                    vdofs |= set(dof for dof in V.GetDofNrs(el) if freedofs[dof])
                blocks.append(vdofs)

            preJpoint = a.mat.CreateBlockSmoother(blocks)
        # preJpoint = SymmetricGS(blockjac)
        else:
            preJpoint = a.mat.CreateSmoother(V.FreeDofs())

        for d in range(V.ndof):
            if V.CouplingType(d) == COUPLING_TYPE.WIREBASKET_DOF:
                vertexdofs[d] = True

        vertexdofs &= V.FreeDofs()

        preCoarse = a.mat.Inverse(vertexdofs, inverse="sparsecholesky")
        preA = MultiplicativePrecond(preJpoint, preCoarse, a.mat)

    else:

        preA = Preconditioner(a, 'bddc')

        a.Assemble()
        b.Assemble()

    preconTimer.Stop()
    mp = BilinearForm(Q)
    mp += SymbolicBFI(p * q)
    preM = Preconditioner(mp, 'local')
    mp.Assemble()

    f = LinearForm(V)
    f += SymbolicLFI((x - 0.5) * v[1])
    f.Assemble()

    g = LinearForm(Q)
    g.Assemble()

    gfu = GridFunction(V, name="u")
    gfp = GridFunction(Q, name="p")
    uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    gfu.components[1].Set(uin, definedon=mesh.Boundaries("inlet"))
    sol = BlockVector([gfu.vec, gfp.vec])
    sol2 = sol.CreateVector()
    sol2[0].data = gfu.vec
    sol2[1].data = gfp.vec

    with TaskManager():#pajetrace=100 * 1000 * 1000):
        bramblePasciakTimer.Start()
        results["nits_bpcg"] = BPCG(a.mat, b.mat, None, f.vec, g.vec, preA, preM, sol2, initialize=False, tol=1e-7,
                                    maxsteps=100000, rel_err=True)
        bramblePasciakTimer.Stop()
        results["time_bpcg"] = bramblePasciakTimer.time

    K = BlockMatrix([[a.mat, b.mat.T], [b.mat, None]])
    C = BlockMatrix([[preA, None], [None, preM]])
    rhs = BlockVector([f.vec, g.vec])
    sol = BlockVector([gfu.vec, gfp.vec])

    with TaskManager():  # pajetrace=100*1000*1000):
        minResTimer.Start()
        tmp, results["nits_minres"] = MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False, tol=1e-7, maxsteps=100000)
        minResTimer.Stop()
        results["time_minres"] = minResTimer.time
    print("BramblePasciakCGMax took", round(bramblePasciakTimer.time, 4), "seconds")
    print("MinRes took", round(minResTimer.time, 4), "seconds")

    return results


V, Q = elements(mesh).bdm(order=2).setup()
#res = spaces_test(V, Q, 2, precon="bddc")
#print({"method": "bdm2", "ndofs": V.ndof + Q.ndof, "precon": "bddc", "vert": mesh.nv, **res})
# spaces_test(V, Q, precon="multi")
#exit(0)

import pandas as pd

data = pd.DataFrame()
element = elements(mesh)
for a in range(4):
    print("#" * 100)
    mesh.Refine()

    print(mesh.nv)
    for e in element_names:
        print("-" * 100, "testing", e)
        for order in [1, 2]:
            V, Q = getattr(element, e)(order).setup()

            for precon in ["bddc", "multi"]:
                result = spaces_test(V, Q, element.order, precon=precon)
                data = data.append({"method": e + str(order), "ndofs": V.ndof + Q.ndof, "precon": precon, "vert": mesh.nv, **result}, ignore_index=True)

            data.to_csv("bpcg_test_all_hdg.csv")
