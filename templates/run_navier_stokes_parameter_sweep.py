from ngsolve import *
from NavierStokesSIMPLE_iterative import *
#import netgen.gui
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions
import pandas as pd
import numpy as np

#ngsglobals.msg_level = 10

def create_mesh(mesh_size):
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
    geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")

    mesh = Mesh(geo.GenerateMesh(maxh=mesh_size))
    mesh.Curve(3)
    return mesh

def solve(mesh, order, gauss_seidel):
    SetHeapSize(100 * 1000 * 1000)
    timestep = 1e-3
    with TaskManager():
        navstokes = NavierStokes(mesh, nu=0.001, order=order, timestep=timestep,
                                 inflow="inlet", outflow="outlet", wall="cyl|wall",
                                 uin=CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0)))

        navstokes.SolveInitial(iterative=True)
        return navstokes

def create_nav_stokes(mesh, order):
    print(f"order: {order}")
    timestep = 1e-3
    with TaskManager():
        navstokes = NavierStokes(mesh, nu=0.001, order=order, timestep=timestep,
                                 inflow="inlet", outflow="outlet", wall="cyl|wall",
                                 uin=CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0)))
        return navstokes

def solve(navstokes, gauss_seidel):
    navstokes.SolveInitial(iterative=True, GS = gauss_seidel)
    return navstokes

mesh_sizes = [2 ** -i for i in range(5, -1, -1)]
orders = range(7, 1, -1)
gauss_seidel_enabled = [True, False]
data_frames = []
SetHeapSize((64 + 128 + 256 + 1024) * 1024 * 1024)
for mesh_size in mesh_sizes:
    print(f"mesh size: {mesh_size}")
    mesh = create_mesh(mesh_size)
    for order in orders:
        navstokes = create_nav_stokes(mesh, order)
        for gauss_seidel in gauss_seidel_enabled:
            print(f"solving h = {mesh_size}, p = {order}, GS = {gauss_seidel}")
            navstokes.SolveInitial(iterative=True, GS = gauss_seidel)
            print(f"solved h = {mesh_size}, p = {order}, GS = {gauss_seidel}")
            data_frames.append(pd.DataFrame({
                                'mesh_size': mesh_size,
                                'order': order,
                                'iterations': navstokes.stokes_bpcg_iterations,
                                'time': navstokes.stokes_bpcg_time,
                                'gauss_seidel_enabled': gauss_seidel},
                                index=[0]))
            Draw(navstokes.velocity,mesh, "velocity")
            Draw(navstokes.pressure,mesh, "pressure")
            visoptions.scalfunction='velocity:0'

data = pd.concat(data_frames, ignore_index=True)
data.to_csv("data.csv")

input("end")
