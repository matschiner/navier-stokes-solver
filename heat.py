import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.ngstd import Timer
from runge_kutta_method import linear_implicit_runge_kutta_step, ImplicitRungeKuttaMethodWeights
from orthonormalization import orthonormalize
from math import pi
import netgen.gui


def sum_of_unit_square_laplace_eigenfunctions(kl):
    temperature = CoefficientFunction(0)
    for k, l in kl:
        temperature += 2 * sin(k * pi * x) * sin(l * pi * y)

    return temperature


def exact_solution(kl, t):
    solution = CoefficientFunction(0)
    for k, l in kl:
        solution += 2 * exp(-(k ** 2 + l ** 2) * pi ** 2 * t) * \
            sin(k * pi * x) * sin(l * pi * y)

    return solution


def solve(initial_temperature, end_time, time_step):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    Draw(mesh)

    space = H1(mesh, order=10, dirichlet='bottom|right|top|left')

    print(space)

    Draw(initial_temperature, mesh, "initial_temperature")

    trial_function = space.TrialFunction()
    test_function = space.TestFunction()

    diffusion_term = grad(trial_function) * grad(test_function)
    diffusion = BilinearForm(space)
    diffusion += diffusion_term * dx

    mass_term = trial_function * test_function
    mass = BilinearForm(space)
    mass += mass_term * dx

    heat = BilinearForm(space)
    heat += mass_term * dx + time_step * diffusion_term * dx

    source_term = 0 * test_function
    source = LinearForm(space)
    source += source_term * dx

    diffusion.Assemble()
    mass.Assemble()
    heat.Assemble()
    source.Assemble()

    temperature = GridFunction(space, "temperature")
    temperature.Set(initial_temperature)
    for i in range(space.ndof):
        if not space.FreeDofs()[i]:
            temperature.vec[i] = 0

    Draw(temperature)

    residual = temperature.vec.CreateVector()
    heat_inverse = heat.mat.Inverse(space.FreeDofs())

    subspace_dimension = 5
    dt = time_step / subspace_dimension
    runge_kutta_weights = ImplicitRungeKuttaMethodWeights(10)
    time = 0
    print(f"time={time}")

    with(TaskManager()):
        while time < end_time:
            time += time_step

            print(f"time={time}")
            timer = Timer("exponential integrators timer")
            timer.Start()
            subspace_basis = [temperature.vec.Copy()]

            initial_condition_norm = Norm(temperature.vec)

            subspace_basis_assembling_timer = Timer(
                "subspace basis assembling")
            subspace_basis_assembling_timer.Start()

            for i in range(1, subspace_dimension):
                residual.data = diffusion.mat * temperature.vec
                temperature.vec.data -= dt * heat_inverse * residual
                subspace_basis.append(temperature.vec.Copy())

            subspace_basis = orthonormalize(subspace_basis)
            subspace_basis_assembling_timer.Stop()

            subspace_matrix_assembling_timer = Timer(
                "subspace matrix assembling")
            subspace_matrix_assembling_timer.Start()
            subspace_diffusion = Matrix(subspace_dimension, subspace_dimension)
            subspace_mass = Matrix(subspace_dimension, subspace_dimension)

            for col in range(subspace_dimension):
                residual.data = diffusion.mat * subspace_basis[col]
                for row in range(subspace_dimension):
                    subspace_diffusion[row, col] = InnerProduct(
                        subspace_basis[row], residual)

                residual.data = mass.mat * subspace_basis[col]
                for row in range(subspace_dimension):
                    subspace_mass[row, col] = InnerProduct(
                        subspace_basis[row], residual)

            subspace_mass_inverse = Matrix(
                subspace_dimension, subspace_dimension)
            subspace_mass.Inverse(subspace_mass_inverse)

            evolution_matrix = -subspace_mass_inverse * subspace_diffusion

            subspace_matrix_assembling_timer.Stop()

            large_timestep_timer = Timer("large timestep")
            large_timestep_timer.Start()

            subspace_temperature = Vector(subspace_dimension)
            subspace_temperature[:] = 0
            subspace_temperature[0] = initial_condition_norm

            next_temperature = linear_implicit_runge_kutta_step(runge_kutta_weights,
                                                                evolution_matrix,
                                                                subspace_temperature,
                                                                time_step)

            temperature.vec[:] = 0
            for i, basis_vector in enumerate(subspace_basis):
                temperature.vec.data += next_temperature[i] * basis_vector

            large_timestep_timer.Stop()
            timer.Stop()
            Redraw()

        return (temperature, mesh, time)


kl = [(1, 1), (2, 1), (1, 3), (3, 3), (2, 3), (4, 5), (5, 2)]
initial_temperature = sum_of_unit_square_laplace_eigenfunctions(kl)
time_steps = np.logspace(-1, -4, num=7).tolist()
end_time = 0.05
error_frames = []
for time_step in time_steps:
    temperature, mesh, time = solve(initial_temperature, end_time, time_step)
    error = sqrt(Integrate(
        (temperature - exact_solution(kl, time)) * (temperature - exact_solution(kl, time)), mesh))
    Draw(temperature, mesh, f"temperature_{time_step}")
    error_frames.append(pd.DataFrame(
        {'time_step': time_step, 'error': error}, index=[0]))

Draw(exact_solution(kl, end_time), mesh, f"exact_solution")
errors = pd.concat(error_frames, ignore_index=True)

errors.to_csv('heat_errors.csv')

input("press to continue")
