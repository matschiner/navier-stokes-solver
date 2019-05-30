from ngsolve import *
from netgen.geom2d import unit_square
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

time_step = 0.01
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

subspace_dimension = 3
dt = time_step / subspace_dimension

time = 0
while True:
    input(f"time = {time}")

    subspace_basis = [temperature.vec.CreateVector()]
    for j, v in enumerate(temperature.vec):
        subspace_basis[0][j] = v

    for i in range(1, subspace_dimension):
        residual.data = diffusion.mat * temperature.vec
        temperature.vec.data -= dt * heat_inverse * residual
        subspace_basis_vector = temperature.vec.CreateVector()
        for j, v in enumerate(temperature.vec):
            subspace_basis_vector[j] = v

        subspace_basis.append(subspace_basis_vector)

    subspace_diffusion = Matrix(subspace_dimension, subspace_dimension)
    subspace_mass = Matrix(subspace_dimension, subspace_dimension)

    for i in range(subspace_dimension):
        for j in range(subspace_dimension):
            residual.data = diffusion.mat * subspace_basis[j]
            subspace_diffusion[(i, j)] = InnerProduct(
                subspace_basis[i], residual)

            residual.data = mass.mat * subspace_basis[j]
            subspace_mass[(i, j)] = InnerProduct(subspace_basis[i], residual)

    print(subspace_diffusion)

    print(subspace_mass)

    subspace_heat = subspace_mass + time_step * subspace_diffusion
    print(subspace_heat)

    subspace_heat_inverse = Matrix(subspace_dimension, subspace_dimension)
    subspace_heat.Inverse(subspace_heat_inverse)
    print(subspace_heat_inverse)

    subspace_temperature = Vector(subspace_dimension)
    subspace_temperature[:] = 0
    subspace_temperature[0] = 1
    subspace_residual = subspace_diffusion.T[0]
    subspace_temperature -= time_step * \
        subspace_heat_inverse * subspace_residual

    temperature.vec[:] = 0
    for i, basis_vector in enumerate(subspace_basis):
        temperature.vec.data += subspace_temperature[i] * basis_vector

    residual.data = diffusion.mat * temperature.vec
    temperature.vec.data -= time_step * heat_inverse * residual
    time += time_step
    Redraw()

input("press to quit\n")
