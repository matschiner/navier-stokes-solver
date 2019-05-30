from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.ngstd import Timer
from runge_kutta_method import linear_runge_kutta_step, ImplicitRungeKuttaMethodWeights
from orthonormalization import orthonormalize
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
runge_kutta_weights = ImplicitRungeKuttaMethodWeights(10)

time = 0

with(TaskManager()):
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

        subspace_mass_inverse = Matrix(subspace_dimension, subspace_dimension)
        subspace_mass.Inverse(subspace_mass_inverse)

        evolution_matrix = -subspace_mass_inverse * subspace_diffusion

        subspace_matrix_assembling_timer.Stop()

        large_timestep_timer = Timer("large timestep")
        large_timestep_timer.Start()

        subspace_temperature = Vector(subspace_dimension)
        subspace_temperature[:] = 0
        subspace_temperature[0] = Norm(temperature.vec)

        subspace_temperature = linear_runge_kutta_step(runge_kutta_weights,
                                                       evolution_matrix,
                                                       subspace_temperature,
                                                       time_step)

        temperature.vec[:] = 0
        for i, basis_vector in enumerate(subspace_basis):
            temperature.vec.data += subspace_temperature[i] * basis_vector

        time += time_step
        large_timestep_timer.Stop()
        timer.Stop()
        Redraw()

input("press to quit\n")
