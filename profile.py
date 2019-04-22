"""Profiling to compare Bramble Pasciak CG method with MinRes method"""

import netgen.gui
from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.ngstd import Timer
from bramble_pasciak_cg import bramble_pasciak_cg
from discretizations import taylor_hood, \
    P1_nonconforming_velocity_constant_pressure, \
    P2_velocity_constant_pressure, \
    P2_velocity_linear_pressure, \
    P2_velocity_with_cubic_bubbles_linear_pressure, \
    mini


def create_mesh(net_width):
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (2, 0.41), bcs=(
        "wall", "outlet", "wall", "inlet"))
    geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
    mesh = Mesh(geo.GenerateMesh(maxh=net_width))
    mesh.Curve(3)
    return mesh


def solve_with_bramble_pasciak_cg(a_matrix, b_matrix, pre_a, pre_schur_complement, gfu, gfp, f, g, tolerance, max_steps):
    sol = BlockVector([gfu, gfp])
    with TaskManager(pajetrace=100 * 1000 * 1000):
        return bramble_pasciak_cg(a_matrix, b_matrix, None, pre_a, pre_schur_complement, f, g, sol,
                                  tolerance=tolerance, max_steps=max_steps)


def solve_with_min_res(a, b, preA, preS, gfu, gfp, f, g, tolerance, max_steps):
    K = BlockMatrix([[a, b.T], [b, None]])
    C = BlockMatrix([[preA, None], [None, preS]])
    rhs = BlockVector([f, g])
    sol = BlockVector([gfu, gfp])
    with TaskManager(pajetrace=100 * 1000 * 1000):
        min_res_timer = Timer("MinRes")
        min_res_timer.Start()
        solvers.MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False,
                       tol=tolerance, maxsteps=max_steps)
        min_res_timer.Stop()
        print("MinRes took", min_res_timer.time, "seconds")
    return sol


def solve(mesh, discretization, solver):
    V, Q = discretization(mesh, velocity_dirichlet='wall|inlet|cyl')

    u, v = V.TnT()
    p, q = Q.TnT()

    a = BilinearForm(V)
    a += SymbolicBFI(grad(u[0]) * grad(v[0]) + grad(u[1]) * grad(v[1]))

    preA = Preconditioner(a, 'bddc')

    b = BilinearForm(trialspace=V, testspace=Q)
    b += SymbolicBFI((grad(u[0])[0] + grad(u[1])[1]) * q)

    a.Assemble()
    b.Assemble()

    mp = BilinearForm(Q)
    mp += SymbolicBFI(p * q)
    preS = Preconditioner(mp, 'local')
    mp.Assemble()

    f = LinearForm(V)
    f += SymbolicLFI((x - 0.5) * v[1])
    f.Assemble()

    g = LinearForm(Q)
    g.Assemble()

    velocity_grid_function = GridFunction(V, name="velocity")
    pressure_grid_function = GridFunction(Q, name="pressure")
    uin_x = CoefficientFunction(1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41))
    velocity_grid_function.components[0].Set(
        uin_x, definedon=mesh.Boundaries("inlet"))
    solver(a.mat, b.mat, preA, preS, velocity_grid_function.vec, pressure_grid_function.vec, f.vec, g.vec,
           tolerance=1e-7, max_steps=10000)
    Draw(CoefficientFunction(
        (velocity_grid_function.components[0], velocity_grid_function.components[1])), mesh, "velocity")
    Draw(pressure_grid_function)
    return (velocity_grid_function, pressure_grid_function)


mesh = create_mesh(net_width=0.1)
discretization = P1_nonconforming_velocity_constant_pressure()
solve(mesh, discretization, solve_with_bramble_pasciak_cg)
solve(mesh, discretization, solve_with_min_res)
input("")
