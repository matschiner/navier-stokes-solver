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

subspace_dimension = 10
dt = time_step / subspace_dimension

time = 0
while True:
    input(f"time = {time}")
    residual.data = diffusion.mat * temperature.vec
    temperature.vec.data -= time_step * heat_inverse * residual
    time += time_step
    Redraw()

input("press to quit\n")
