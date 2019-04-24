from ngsolve import *

from solvers.krylovspace import *

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

mesh = Mesh(geo.GenerateMesh(maxh=0.01))

mesh.Curve(3)


def spaces_test(precon="bddc"):
    results = {}
    order = 3
    V1 = HDiv(mesh, order=order, dirichlet="wall|inlet|cyl", RT=True)
    V2 = HCurlDiv(mesh, order=order)
    Q = L2(mesh, order=order)
    X = FESpace([V1, V2, Q])
    results = {}

    u, v = V1.TnT()
    sigma, tau = V2.TnT()
    p, q = Q.TnT()

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

    return results, V1.ndof + V2.ndof + Q.ndof


# spaces_test(V, Q)
# spaces_test(V, Q, precon="multi")
# exit(0)

import pandas as pd

data = pd.DataFrame()
for a in range(3):
    print("#" * 100, a)
    mesh.Refine()

    print(mesh.nv)

    result, ndofs = spaces_test()
    data = data.append({"method": "curldiv", "ndofs": ndofs, "precon": "bddc", "vert": mesh.nv, **result}, ignore_index=True)
    data.to_csv("bpcg_test_all_nc.csv")
