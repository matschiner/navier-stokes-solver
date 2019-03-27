from ngsolve import *

# viscosity
nu = 0.001

# timestepping parameters
tau = 0.001
tend = 10

from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
mesh = Mesh( geo.GenerateMesh(maxh=0.08))

mesh.Curve(3)

V = VectorH1(mesh, order=3, dirichlet="wall|inlet|cyl")
Q = H1(mesh, order=2)

u,v = V.TnT()
p,q = Q.TnT()

a = BilinearForm(V)
a += SymbolicBFI(InnerProduct(grad(u),grad(v)))

b = BilinearForm(trialspace=V, testspace=Q)
b += SymbolicBFI(div(u)*q)

a.Assemble()
b.Assemble()

mp = BilinearForm(Q)
mp += SymbolicBFI(p*q)
mp.Assemble()

f = LinearForm(V)
f += SymbolicLFI( CoefficientFunction( (0,x-0.5)) * v)
f.Assemble()

g = LinearForm(Q)
g.Assemble()

gfu = GridFunction(V, name="u")
gfp = GridFunction(Q, name="p")
uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.Set(uin, definedon=mesh.Boundaries("inlet"))

K = BlockMatrix( [ [a.mat, b.mat.T], [b.mat, None] ] )
C = BlockMatrix( [ [a.mat.Inverse(V.FreeDofs()), None], [None, mp.mat.Inverse()] ] )

rhs = BlockVector ( [f.vec, g.vec] )
sol = BlockVector( [gfu.vec, gfp.vec] )

solvers.MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False)

Draw(Norm(gfu.components[0]), mesh, "vel")
