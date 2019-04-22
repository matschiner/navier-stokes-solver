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
