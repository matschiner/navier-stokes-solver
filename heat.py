from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.ngstd import Timer
import netgen.gui

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
Draw(mesh)

space = H1(mesh, order=2, dirichlet='bottom|right|top|left')

initial_temperature = 10 * x * (1 - x) * y * (1 - y)

Draw(initial_temperature, mesh, "initial_temperature")

trial_function = space.TrialFunction()
test_function = space.TestFunction()

diffusion_term = grad(trial_function) * grad(test_function)
diffusion = BilinearForm(space)
diffusion += diffusion_term * dx

mass_term = trial_function * test_function
mass = BilinearForm(space)
mass += mass_term * dx

time_step = 0.001
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
Draw(temperature)

residual = temperature.vec.CreateVector()
residual.data = source.vec - diffusion.mat * temperature.vec
heat_inverse = heat.mat.Inverse(space.FreeDofs())

subspace_dimension = 10
dt = time_step / subspace_dimension

time = 0


def orthonormalize(basis, tries=3):
    orthonormalize_timer = Timer("orthonormalization")
    orthonormalize_timer.Start()
    for tries in range(tries):
        for j in range(len(basis)):
            for i in range(j):
                basis[j].data -= InnerProduct(basis[i], basis[j]) / \
                    InnerProduct(basis[i], basis[i]) * basis[i]
            basis[j].data = 1 / Norm(basis[j]) * basis[j]
    orthonormalize_timer.Stop()

    return basis


with(TaskManager(pajetrace=100 * 1000 * 1000)):
    while True:
        input(f"time = {time}")

        timer = Timer("exponential integrators timer")
        timer.Start()
        subspace_basis = [temperature.vec.Copy()]

        subspace_basis_assembling_timer = Timer("subspace basis assembling")
        subspace_basis_assembling_timer.Start()
        for i in range(1, subspace_dimension):
            residual.data = diffusion.mat * temperature.vec
            temperature.vec.data -= dt * heat_inverse * residual
            subspace_basis.append(temperature.vec.Copy())
        subspace_basis_assembling_timer.Stop()

        subspace_basis = orthonormalize(subspace_basis)

        subspace_matrix_assembling_timer = Timer("subspace matrix assembling")
        subspace_matrix_assembling_timer.Start()
        subspace_diffusion = Matrix(subspace_dimension, subspace_dimension)
        subspace_mass = Matrix(subspace_dimension, subspace_dimension)

        for i in range(subspace_dimension):
            for j in range(subspace_dimension):
                residual.data = diffusion.mat * subspace_basis[j]
                subspace_diffusion[(i, j)] = InnerProduct(
                    subspace_basis[i], residual)

                residual.data = mass.mat * subspace_basis[j]
                subspace_mass[(i, j)] = InnerProduct(
                    subspace_basis[i], residual)

        subspace_heat = subspace_mass + time_step * subspace_diffusion

        subspace_heat_inverse = Matrix(subspace_dimension, subspace_dimension)
        subspace_heat.Inverse(subspace_heat_inverse)

        subspace_matrix_assembling_timer.Stop()

        large_timestep_timer = Timer("large timestep")
        large_timestep_timer.Start()

        subspace_temperature = Vector(subspace_dimension)
        subspace_temperature[:] = 0
        subspace_temperature[0] = Norm(temperature.vec)
        subspace_residual = subspace_diffusion.T[0]
        subspace_temperature -= time_step * subspace_heat_inverse * subspace_residual

        temperature.vec[:] = 0
        for i, basis_vector in enumerate(subspace_basis):
            temperature.vec.data += subspace_temperature[i] * basis_vector

        residual.data = diffusion.mat * temperature.vec
        temperature.vec.data -= time_step * heat_inverse * residual
        time += time_step
        large_timestep_timer.Stop()
        timer.Stop()
        Redraw()

input("press to quit\n")
