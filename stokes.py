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
maxh = 0.01
mesh = Mesh(geo.GenerateMesh(maxh=maxh))

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


element_names = [a for a in filter(lambda x: "__" not in x and x not in ["setup", "mesh"], dir(elements(mesh)))]


def spaces_test(V, Q):
    results = {}

    u, v = V.TnT()
    p, q = Q.TnT()

    a = BilinearForm(V)
    a += SymbolicBFI(InnerProduct(grad(u), grad(v)))

    b = BilinearForm(trialspace=V, testspace=Q)
    b += SymbolicBFI(div(u) * q)

    # print("multiplicateive Precond")
    # a.Assemble()
    # b.Assemble()
    #
    # preJpoint = a.mat.CreateSmoother(V.FreeDofs())
    #
    # vertexdofs = BitArray(V.ndof)
    # vertexdofs[:] = False
    #
    # for v in mesh.vertices:
    #    for d in V.GetDofNrs(v):
    #        vertexdofs[d] = True
    #
    # vertexdofs &= V.FreeDofs()
    #
    # coarsepre = a.mat.Inverse(vertexdofs)
    #
    # preA = MultiplicativePrecond(preJpoint, coarsepre, a.mat)

    preA = Preconditioner(a, 'bddc')
    a.Assemble()
    b.Assemble()

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
    gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
    sol = BlockVector([gfu.vec, gfp.vec])
    sol2 = sol.CreateVector()
    sol2[0].data = gfu.vec
    sol2[1].data = gfp.vec

    with TaskManager():  # pajetrace=100*1000*1000):
        bramblePasciakTimer = Timer("BramblePasciakCG")
        bramblePasciakTimer.Start()
        results["nits_bpcg"] = BPCG_Max(a.mat, b.mat, None, f.vec, g.vec, preA, preM, sol2, initialize=False, tol=1e-7, maxsteps=100000, rel_err=True)
        bramblePasciakTimer.Stop()
        results["time_bpcg"] = bramblePasciakTimer.time

    gfu = GridFunction(V, name="u")
    gfp = GridFunction(Q, name="p")
    uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
    sol = BlockVector([gfu.vec, gfp.vec])
    K = BlockMatrix([[a.mat, b.mat.T], [b.mat, None]])
    C = BlockMatrix([[preA, None], [None, preM]])
    rhs = BlockVector([f.vec, g.vec])
    sol = BlockVector([gfu.vec, gfp.vec])
    with TaskManager():  # pajetrace=100*1000*1000):
        minResTimer = Timer("MinRes")
        minResTimer.Start()
        tmp, results["nits_minres"] = MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False, tol=1e-7, maxsteps=100000)
        minResTimer.Stop()
        results["time_minres"] = minResTimer.time
        # end taskman
    if V.ndof + Q.ndof < 602560:
        try:
            XV = VectorH1(mesh, order=2, dirichlet="wall|inlet|cyl")
            V.SetOrder(TRIG, 3)
            V.Update()
            Q = L2(mesh, order=1)
            X = FESpace([V, Q])
            u, p = X.TrialFunction()
            v, q = X.TestFunction()
            a = BilinearForm(X)
            a += SymbolicBFI(InnerProduct(grad(u), grad(v)) + div(u) * q + div(v) * p)
            a.Assemble()
            gfu = GridFunction(X)
            uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
            gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
            directTimer = Timer("direct")
            directTimer.Start()
            res = gfu.vec.CreateVector()
            with TaskManager():
                res.data = -a.mat * gfu.vec
                inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
                gfu.vec.data += inv * res
                directTimer.Stop()
            Draw(Norm(gfu.components[0]), mesh, "vel")
            print("direct solver took", round(directTimer.time, 4), "seconds")
            results["time_direct"] = directTimer.time
        except Exception as e:
            print(str(e))

    print("BramblePasciakCGMax took", round(bramblePasciakTimer.time, 4), "seconds")
    # print("BramblePasciakCG took", bramblePasciakTimer2.time, "seconds")
    print("MinRes took", round(minResTimer.time, 4), "seconds")

    return results


V, Q = elements(mesh).bubbled().setup()
# spaces_test()
import pandas as pd

data = pd.DataFrame()
element = elements(mesh)
for a in range(3):
    print("#" * 100, a)
    mesh.Refine()
    V.Update()
    Q.Update()
    print(mesh.nv)
    for e in element_names:
        print("testing", e)
        V, Q = getattr(element, e)().setup()
        result = {}
        result = spaces_test(V, Q)
        data = data.append({"method": e, "ndofs": V.ndof + Q.ndof, "vert": mesh.nv, **result}, ignore_index=True)
        data.to_csv("bpcg_test.csv")
