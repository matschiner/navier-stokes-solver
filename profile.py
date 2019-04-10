"""Profiling to compare Bramble Pasciak CG method with MinRes method"""

from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.ngstd import Timer
from bramble_pasciak_cg import bramble_pasciak_cg

def create_mesh():
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
    geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
    mesh = Mesh(geo.GenerateMesh(maxh=0.003))
    mesh.Curve(3)
    return mesh

def solve_with_bramble_pasciak_cg(a_matrix, b_matrix, pre_a, pre_schur_complement, gfu, gfp, f, g, tolerance, max_steps):
    sol = BlockVector([gfu, gfp])
    with TaskManager(pajetrace=100*1000*1000):
        bramble_pasciak_cg(a_matrix, b_matrix, None, pre_a, pre_schur_complement, f, g, sol, \
                           tolerance=tolerance, max_steps=max_steps)

def solve_with_max_bramble_pasciak_cg(a_matrix, b_matrix, pre_a, pre_schur_complement, gfu, gfp, f, g, tolerance, max_steps):
    sol = BlockVector([gfu, gfp])
    with TaskManager(pajetrace=100*1000*1000):
        BPCG_Max(a_matrix, b_matrix, None, f, g, pre_a, pre_schur_complement, sol, \
                           tol=tolerance, maxsteps=max_steps)

def solve_with_min_res(a, b, preA, preS, gfu, gfp, f, g, tolerance, max_steps):
    sol = BlockVector([gfu, gfp])
    K = BlockMatrix([[a, b.T], [b, None]])
    C = BlockMatrix([[preA, None], [None, preS]])
    rhs = BlockVector([f, g])
    sol = BlockVector([gfu, gfp])
    with TaskManager(pajetrace=100 * 1000 * 1000):
        min_res_timer = Timer("MinRes")
        min_res_timer.Start()
        solvers.MinRes(mat=K, pre=C, rhs=rhs, sol=sol, \
                       initialize=False, tol=tolerance, maxsteps=max_steps)
        min_res_timer.Stop()
        print("MinRes took", min_res_timer.time, "seconds")

def solve(solver):
    mesh = create_mesh()
    V = VectorH1(mesh, order=3, dirichlet="wall|inlet|cyl")
    Q = H1(mesh, order=2)

    u, v = V.TnT()
    p, q = Q.TnT()

    a = BilinearForm(V)
    a += SymbolicBFI(InnerProduct(grad(u), grad(v)))

    preA = Preconditioner(a, 'bddc')

    b = BilinearForm(trialspace=V, testspace=Q)
    b += SymbolicBFI(div(u)*q)

    a.Assemble()
    b.Assemble()

    mp = BilinearForm(Q)
    mp += SymbolicBFI(p * q)
    preS = Preconditioner(mp, 'local')
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
    solver(a.mat, b.mat, preA, preS, gfu.vec, gfp.vec, f.vec, g.vec, \
           tolerance=1e-7, max_steps=1000)
    Draw(gfu.components[0])
    Draw(gfu.components[1])
    Draw(gfp)
    return (gfu, gfp)

solve(solve_with_bramble_pasciak_cg)
solve(solve_with_min_res)
