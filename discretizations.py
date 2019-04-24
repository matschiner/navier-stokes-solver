"""different discretizations for navier stokes equation"""

from ngsolve import *


def taylor_hood(order):
    def discretization(mesh, velocity_dirichlet):
        velocity_space = H1(mesh, order=order, dirichlet=velocity_dirichlet)
        return (FESpace([velocity_space, velocity_space]),
                H1(mesh, order=order - 1))
    return discretization


def P1_nonconforming_velocity_constant_pressure():
    def discretization(mesh, velocity_dirichlet):
        velocity_space = FESpace(
            'nonconforming', mesh, order=1, dirichlet=velocity_dirichlet)
        return (FESpace([velocity_space, velocity_space]),
                L2(mesh, order=0))
    return discretization


def P2_velocity_constant_pressure():
    def discretization(mesh, velocity_dirichlet):
        velocity_space = H1(mesh, order=2, dirichlet=velocity_dirichlet)
        return (FESpace([velocity_space, velocity_space]),
                L2(mesh, order=0))
    return discretization


def P2_velocity_linear_pressure():
    def discretization(mesh, velocity_dirichlet):
        velocity_space = H1(mesh, order=2, dirichlet=velocity_dirichlet)
        return (FESpace([velocity_space, velocity_space]),
                L2(mesh, order=1))
    return discretization


def P2_velocity_with_cubic_bubbles_linear_pressure():
    def discretization(mesh, velocity_dirichlet):
        velocity_space = H1(mesh, order=2, dirichlet=velocity_dirichlet)
        velocity_space.SetOrder(TRIG, 3)
        velocity_space.Update()
        return (FESpace([velocity_space, velocity_space]),
                L2(mesh, order=1))
    return discretization


def mini():
    def discretization(mesh, velocity_dirichlet):
        velocity_space = H1(mesh, order=1, dirichlet=velocity_dirichlet)
        velocity_space.SetOrder(TRIG, 3)
        velocity_space.Update()
        return (FESpace([velocity_space, velocity_space]),
                H1(mesh, order=1))
    return discretization


def bdm_hybrid(order, penalty, hodivfree=False):
    def discretization(mesh, velocity_dirichlet):
        velocity_space = HDiv(
            mesh, order=order, dirichlet=velocity_dirichlet, hodivfree=hodivfree)
        velocity_facet_space = VectorFacet(
            mesh, order=order, dirichlet=velocity_dirichlet)
        pressure_space = L2(mesh, order=0 if hodivfree else (order - 1))
        return (FESpace([velocity_space, velocity_facet_space]), pressure_space)
    return discretization


def rt_hybrid(order, penalty, hodivfree=False):
    def discretization(mesh, velocity_dirichlet):
        velocity_space = HDiv(
            mesh, order=order, dirichlet=velocity_dirichlet, hodivfree=hodivfree, RT=True)
        velocity_facet_space = VectorFacet(
            mesh, order=order, dirichlet=velocity_dirichlet)
        pressure_space = L2(mesh, order=0 if hodivfree else (order - 1))
        return (FESpace([velocity_space, velocity_facet_space]), pressure_space)
    return discretization


def hcurldiv(order, raviart_thomas=True):
    def discretization(mesh, velocity_dirichlet, velocity_neumann):
        velocity_space = HDiv(mesh, order=order,
                              dirichlet=velocity_dirichlet, RT=raviart_thomas)
        V2 = HCurlDiv(mesh, order=order, dirichlet=velocity_neumann)
        pressure_space = L2(mesh, order=order)
        return (velocity_space, V2, pressure_space)
    return discretization
