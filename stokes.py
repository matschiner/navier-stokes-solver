from ngsolve.la import *
from solvers.krylovspace import *
from solvers.bramblepasciak import BramblePasciakCG as BPCG_Max
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

    def taylor_hood3(self, order=3):
        self.v1, self.v2, self.setup_lambda = (
            VectorH1(self.mesh, order=order, dirichlet="wall|inlet|cyl"),
            H1(self.mesh, order=order - 1),
            lambda v1, v2: 1,
        )
        return self

    def taylor_hood(self, order=2):
        self.v1, self.v2, self.setup_lambda = (
            VectorH1(self.mesh, order=order, dirichlet="wall|inlet|cyl"),
            H1(self.mesh, order=order - 1),
            lambda v1, v2: 1,
        )
        return self

    def bubbled(self, order=2):
        self.v1, self.v2, self.setup_lambda = (
            VectorH1(self.mesh, order=order, dirichlet="wall|inlet|cyl"),
            L2(self.mesh, order=order - 1),
            lambda v1, v2: (v1.SetOrder(TRIG, 3), v1.Update()),
        )
        return self

    def mini(self, order=1):
        self.v1, self.v2, self.setup_lambda = (
            VectorH1(self.mesh, order=order, dirichlet="wall|inlet|cyl"),
            H1(self.mesh, order=order),
            lambda v1, v2: (v1.SetOrder(TRIG, 3), v1.Update()),
        )
        return self

    def p2p0(self, order=2):
        self.v1, self.v2, self.setup_lambda = (
            VectorH1(self.mesh, order=order, dirichlet="wall|inlet|cyl"),
            L2(self.mesh, order=order - 2),
            lambda v1, v2: 1,
        )

        return self

    def setup(self):
        self.setup_lambda(self.v1, self.v2)
        return self.v1, self.v2


element_names = [a for a in filter(lambda x: "__" not in x and x not in ["setup", "mesh", "taylor_hood3"], dir(elements(mesh)))]


def spaces_test(V, Q, precon="bddc"):
    results = {}

    u, v = V.TnT()
    p, q = Q.TnT()

    a = BilinearForm(V)
    a += SymbolicBFI(InnerProduct(grad(u), grad(v)))

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
        preA = Preconditioner(a, 'bddc')
        a.Assemble()
        b.Assemble()

    preconTimer.Stop()
    result["preconTime"] = preconTimer.time

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
    uin = CoefficientFunction(1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41))
    gfu.components[1].Set(uin, definedon=mesh.Boundaries("inlet"))
    sol = BlockVector([gfu.vec, gfp.vec])

    sol2 = sol.CreateVector()
    sol2[0].data = gfu.vec
    sol2[1].data = gfp.vec

    with TaskManager():  # pajetrace=100*1000*1000):
        bramblePasciakTimer.Start()
        results["nits_bpcg"] = BPCG_Max(a.mat, b.mat, None, f.vec, g.vec, preA, preM, sol2, initialize=False, tol=1e-7, maxsteps=100000, rel_err=True)
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

    # solve direct only if reasonable
    # if V.ndof + Q.ndof < 802560:
    #    try:
    #        XV = VectorH1(mesh, order=2, dirichlet="wall|inlet|cyl")
    #        V.SetOrder(TRIG, 3)
    #        V.Update()
    #        Q = L2(mesh, order=1)
    #        X = FESpace([V, Q])
    #        u, p = X.TrialFunction()
    #        v, q = X.TestFunction()
    #        a = BilinearForm(X)
    #        a += SymbolicBFI(InnerProduct(grad(u[0]), grad(v[0])) + InnerProduct(grad(u[1]), grad(v[1])))
    #        a += SymbolicBFI((grad(u[0])[0] + grad(u[1])[1]) * q)
    #        a += SymbolicBFI((grad(v[0])[0] + grad(v[1])[1]) * p)
    #
    #        #a += SymbolicBFI(InnerProduct(grad(u), grad(v)) + div(u) * q + div(v) * p)
    #        a.Assemble()
    #        gfu = GridFunction(X)
    #        uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    #        gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
    #        directTimer = Timer("direct")
    #        directTimer.Start()
    #        res = gfu.vec.CreateVector()
    #        with TaskManager():
    #            res.data = -a.mat * gfu.vec
    #            inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    #            gfu.vec.data += inv * res
    #            directTimer.Stop()
    #        Draw(Norm(gfu.components[0]), mesh, "vel")
    #        print("direct solver took", round(directTimer.time, 4), "seconds")
    #        results["time_direct"] = directTimer.time
    #    except Exception as e:
    #        print(str(e))
    #
    print("BramblePasciakCGMax took", round(bramblePasciakTimer.time, 4), "seconds")
    print("MinRes took", round(minResTimer.time, 4), "seconds")

    return results


V, Q = elements(mesh).bubbled().setup()
# spaces_test(V, Q)
# spaces_test(V, Q,precon="multi")
# exit(0)
import pandas as pd

data = pd.DataFrame()
element = elements(mesh)
for a in range(4):
    print("#" * 100, a)
    mesh.Refine()
    V.Update()
    Q.Update()
    print(mesh.nv)
    for e in element_names:
        print("-" * 100, "testing", e)
        V, Q = getattr(element, e)().setup()
        result = {}
        for precon_name in ["bddc", "multi"]:
            result = spaces_test(V, Q, precon=precon_name)
            data = data.append({"method": e, "ndofs": V.ndof + Q.ndof, "precon": precon_name, "vert": mesh.nv, **result}, ignore_index=True)
            data.to_csv("bpcg_test_precons.csv")
