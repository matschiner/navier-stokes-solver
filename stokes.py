from ngsolve.la import *
from ngsolve.la import EigenValues_Preconditioner
from math import sqrt
from ngsolve import *
from ngsolve.ngstd import Timer
from krylovspace import *
from solvers import BramblePasciakCG as BPCG_Max

# ngsglobals.msg_level = 0

# viscosity
nu = 0.001

# timestepping parameters
tau = 0.001
tend = 10

# fem size
maxh = 0.00625

from netgen.geom2d import SplineGeometry

geo = SplineGeometry()
geo.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")

mesh = Mesh(geo.GenerateMesh(maxh=maxh))

mesh.Curve(3)

V = VectorH1(mesh, order=3, dirichlet="wall|inlet|cyl")
Q = H1(mesh, order=2)

u, v = V.TnT()
p, q = Q.TnT()

a = BilinearForm(V)
a += SymbolicBFI(InnerProduct(grad(u), grad(v)))

preA = Preconditioner(a, 'bddc')

b = BilinearForm(trialspace=V, testspace=Q)
b += SymbolicBFI(div(u) * q)

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
    scf = BPCG_Max(a.mat, b.mat, None, f.vec, g.vec, preA, preM, sol2, initialize=False, tol=1e-12, maxsteps=100000,
                   rel_err=True)
    bramblePasciakTimer.Stop()

    bramblePasciakTimer2 = Timer("BramblePasciakCG2")
    bramblePasciakTimer2.Start()
    # BramblePasciakCG(a.mat, b.mat, None, f.vec, g.vec, preA, preM, sol, initialize=False, tol=1e-7, maxsteps=1000, scf=scf)
    bramblePasciakTimer2.Stop()

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
    MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False, tol=1e-12, maxsteps=100000)
    minResTimer.Stop()

sol.data-=sol2
print("diff",Norm(sol))

    # end taskman
if maxh > 0.01:
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

print("BramblePasciakCGMax took", round(bramblePasciakTimer.time, 4), "seconds")
# print("BramblePasciakCG took", bramblePasciakTimer2.time, "seconds")
print("MinRes took", round(minResTimer.time, 4), "seconds")
