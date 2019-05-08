"""Profiling to compare Bramble Pasciak CG method with MinRes method"""

import pandas as pd
import sys
import netgen.gui
from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.ngstd import Timer
from minres import MinRes
from bramble_pasciak_cg import bramble_pasciak_cg
from discretizations import taylor_hood, \
    P1_nonconforming_velocity_constant_pressure, \
    P2_velocity_constant_pressure, \
    P2_velocity_linear_pressure, \
    P2_velocity_with_cubic_bubbles_linear_pressure, \
    mini, \
    bdm_hybrid, \
    rt_hybrid, \
    hcurldiv


def create_mesh(mesh_size):
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (2, 0.41), bcs=(
        "wall", "outlet", "wall", "inlet"))
    geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
    mesh = Mesh(geo.GenerateMesh(maxh=mesh_size))
    mesh.Curve(3)
    return mesh


def solve_with_bramble_pasciak_cg(a_matrix, b_matrix, pre_a, pre_schur_complement, gfu, gfp, f, g, tolerance, max_steps):
    sol = BlockVector([gfu, gfp])
    bramble_pasciak_cg_timer = Timer("BramblePasciakCG")
    bramble_pasciak_cg_timer.Start()
    (solution, errors) = bramble_pasciak_cg(a_matrix, b_matrix, None, pre_a, pre_schur_complement, f, g, sol,
                                            tolerance=tolerance, max_steps=max_steps)
    bramble_pasciak_cg_timer.Stop()
    print("Bramble Pasciak CG took",
          bramble_pasciak_cg_timer.time, "seconds")
    return (solution, errors, bramble_pasciak_cg_timer.time)


def solve_with_min_res(a, b, preA, preS, gfu, gfp, f, g, tolerance, max_steps):
    K = BlockMatrix([[a, b.T], [b, None]])
    C = BlockMatrix([[preA, None], [None, preS]])
    rhs = BlockVector([f, g])
    sol = BlockVector([gfu, gfp])

    min_res_timer = Timer("MinRes")
    min_res_timer.Start()
    (solution, errors) = MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False,
                                tol=tolerance, maxsteps=max_steps)
    min_res_timer.Stop()
    print("MinRes took", min_res_timer.time, "seconds")
    return (solution, errors, min_res_timer.time)


def create_iterative_solver_factory(solver, a_pre, schur_complement_pre, tolerance, max_steps):
    def create_iterative_solver(space, a, b, m):
        pre_a = Preconditioner(a, a_pre)
        pre_schur_complement = Preconditioner(m, schur_complement_pre)

        def solve(a_matrix, b_matrix, gfu, gfp, f, g):
            return solver(a_matrix, b_matrix, pre_a, pre_schur_complement, gfu, gfp, f, g, tolerance, max_steps)

        return solve
    return create_iterative_solver


def solve(mesh, discretization, solver_factory):
    V, Q = discretization(mesh, velocity_dirichlet='wall|inlet|cyl')

    u, v = V.TnT()
    p, q = Q.TnT()

    a = BilinearForm(V)
    a += SymbolicBFI(grad(u[0]) * grad(v[0]) + grad(u[1]) * grad(v[1]))

    b = BilinearForm(trialspace=V, testspace=Q)
    b += SymbolicBFI((grad(u[0])[0] + grad(u[1])[1]) * q)

    mp = BilinearForm(Q)
    mp += SymbolicBFI(p * q)

    solver = solver_factory(FESpace([V, Q]), a, b, mp)

    a.Assemble()
    b.Assemble()
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

    solution, errors, time = solver(
        a.mat, b.mat, velocity_grid_function.vec, pressure_grid_function.vec, f.vec, g.vec)
    Draw(CoefficientFunction((velocity_grid_function.components[0],
                              velocity_grid_function.components[1])), mesh, "velocity")
    Draw(pressure_grid_function)
    ndofs = V.ndof + Q.ndof
    return (velocity_grid_function, pressure_grid_function, errors, time, ndofs)


def solve_hybrid(mesh, discretization, solver_factory):
    alpha = 10
    order = 2
    hodivfree = False

    V, Q = discretization(mesh, velocity_dirichlet='wall|inlet|cyl')
    (u, uhat), (v, vhat) = V.TnT()
    p, q = Q.TnT()

    gradu = CoefficientFunction((grad(u),), dims=(2, 2)).trans
    gradv = CoefficientFunction((grad(v),), dims=(2, 2)).trans

    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size

    def tang(vec):
        return vec - (vec * n) * n

    a = BilinearForm(V, symmetric=True)
    a += SymbolicBFI(InnerProduct(gradu, gradv))
    a += SymbolicBFI(InnerProduct(gradu * n, tang(vhat - v)),
                     element_boundary=True)
    a += SymbolicBFI(InnerProduct(gradv * n, tang(uhat - u)),
                     element_boundary=True)
    a += SymbolicBFI(alpha * order * order / h
                     * InnerProduct(tang(vhat - v), tang(uhat - u)), element_boundary=True)

    b = BilinearForm(trialspace=V, testspace=Q)
    b += SymbolicBFI(div(u) * q)

    mp = BilinearForm(Q)
    mp += SymbolicBFI(p * q)

    solver = solver_factory(FESpace([V, Q]), a, b, mp)

    a.Assemble()
    b.Assemble()
    mp.Assemble()

    f = LinearForm(V)
    f += SymbolicLFI((x - 0.5) * v[1])
    f.Assemble()

    g = LinearForm(Q)
    g.Assemble()

    velocity_grid_function = GridFunction(V, name="velocity")
    pressure_grid_function = GridFunction(Q, name="pressure")
    uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    velocity_grid_function.components[0].Set(
        uin, definedon=mesh.Boundaries("inlet"))

    solution, errors, time = solver(
        a.mat, b.mat, velocity_grid_function.vec, pressure_grid_function.vec, f.vec, g.vec)
    Draw(velocity_grid_function.components[0], mesh, "velocity")
    Draw(velocity_grid_function.components[1], mesh, "velocity_facets")
    Draw(pressure_grid_function)
    ndofs = V.ndof + Q.ndof
    return (velocity_grid_function, pressure_grid_function, errors, time, ndofs)


def solve_hcurldiv(mesh, discretization, solver_factory):
    V, S, Q = discretization(
        mesh, velocity_dirichlet='wall|inlet|cyl', velocity_neumann='outlet')

    X = FESpace([V, S, Q])
    (u, sigma, p), (v, tau, q) = X.TnT()

    n = specialcf.normal(mesh.dim)

    a = BilinearForm(X, symmetric=True)
    a += SymbolicBFI(InnerProduct(sigma, tau))
    a += SymbolicBFI(div(sigma) * v + div(tau) * u)
    a += SymbolicBFI(-(sigma * n) * n * (v * n) - (tau * n) *
                     n * (u * n), element_boundary=True)
    a += SymbolicBFI(div(u) * q + div(v) * p)

    a.Assemble()

    f = LinearForm(X)
    f += SymbolicLFI(CoefficientFunction((0, x - 0.5)) * v)
    f.Assemble()

    grid_function = GridFunction(X)
    uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
    grid_function.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

    direct_timer = Timer("Direct Solver")
    direct_timer.Start()
    res = grid_function.vec.CreateVector()
    res.data = f.vec - a.mat * grid_function.vec
    inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
    grid_function.vec.data += inv * res
    direct_timer.Stop()

    velocity = CoefficientFunction(grid_function.components[0])
    pressure = CoefficientFunction(grid_function.components[2])

    Draw(velocity, mesh, "velocity")
    Draw(pressure, mesh, "pressure")
    ndofs = X.ndof
    return (velocity, pressure, [], direct_timer.time, ndofs)


def profiling_enabled():
    return '-p' in sys.argv[1:]


def data_file():
    return next((argument for argument in sys.argv[1:]
                 if not argument.startswith('-')), "errors.csv")


def run(mesh_sizes, methods, solver_factories, data_file, profiling_enabled, profiling_file_size=100 * 1000 * 1000):
    error_frames = []
    for mesh_size in mesh_sizes:
        mesh = create_mesh(mesh_size=mesh_size)
        for method_name, method_map in methods.items():
            solve_method = method_map['solve']
            discretizations = method_map['discretizations']
            for discretization_name, (discretization, order) in discretizations.items():
                for solver_name, solver in solver_factories.items():
                    message = ", ".join(
                        [discretization_name, solver_name, "h=" + str(mesh_size)])
                    print("solving with", message)
                    with TaskManager(pajetrace=profiling_file_size) if profiling_enabled else TaskManager():
                        solution, _, errors, solver_time, ndofs = solve_method(
                            mesh, discretization, solver)
                    print("\n")

                    error_frames.append(pd.DataFrame({
                        'mesh_size': mesh_size,
                        'discretization': discretization_name,
                        'order': order,
                        'solver': solver_name,
                        'iteration': range(len(errors)),
                        'error': errors,
                        'solver_time': solver_time,
                        'nvertices': mesh.nv,
                        'nedges': mesh.nedge,
                        'nfaces': mesh.nface,
                        'nfacets': mesh.nfacet,
                        'nelements': mesh.ne,
                        'ndofs': ndofs,
                        'method': method_name
                    }))

    data = pd.concat(error_frames, ignore_index=True)
    data.to_csv(data_file)


mesh_sizes = [0.1]  # , 0.05, 0.025, 0.01]
methods = {'mixed': {'solve': solve,
                     'discretizations': {
                         # "P1nc, P0": P1_nonconforming_velocity_constant_pressure(),
                         # "mini": mini(),
                         # "P2, P0": P2_velocity_constant_pressure(),
                         # "P2+, P1": P2_velocity_with_cubic_bubbles_linear_pressure(),
                         # "taylor hood 2": taylor_hood(2),
                         # "taylor hood 3": taylor_hood(3),
                     }},
           'hybrid_dg': {'solve': solve_hybrid,
                         'discretizations': {
                             # "HDG BDM 1": bdm_hybrid(1, 10),
                             "HDG BDM 2": bdm_hybrid(2, 10),
                             # "HDG RT 0": rt_hybrid(0, 10),
                             # "HDG RT 1": rt_hybrid(1, 10),
                             # "HDG RT 2": rt_hybrid(2, 10)
                         }},
           'mcs': {'solve': solve_hcurldiv,
                   'discretizations': {
                       # "MCS RT 0": hcurldiv(0),
                       # "MCS RT 1": hcurldiv(1),
                       # "MCS RT 2": hcurldiv(2)
                   }}}
solver_factories = {
    "bramble pasciak cg": create_iterative_solver_factory(solve_with_bramble_pasciak_cg,
                                                          a_pre='bddc', schur_complement_pre='local',
                                                          tolerance=1e-7, max_steps=10000),
    # "minres": create_iterative_solver_factory(solve_with_min_res,
    #                                          a_pre='bddc', schur_complement_pre='local',
    #                                          tolerance=1e-7, max_steps=10000)
}

print("profiling_enabled:", profiling_enabled())
print("data file:", data_file())
run(mesh_sizes, methods, solver_factories,
    data_file(), profiling_enabled())

input("press to quit")
