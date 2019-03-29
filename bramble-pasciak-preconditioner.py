from ngsolve.la import *
from ngsolve.la import EigenValues_Preconditioner
from math import sqrt
from ngsolve import *
from ngsolve.ngstd import Timer

class ScaledPreconditioner(BaseMatrix):
    def __init__ (self, factor, matrix, pre):
        super(ScaledPreconditioner, self).__init__()
        self.factor = factor
        self.matrix = matrix
        self.pre = pre
    def MultAdd(self, s, x, y):
        y.data = self.factor * s * self.pre * x
    def MultTransAdd(self, s, x, y):
        # pre is assumed to be symmetric therefore leave out .T
        y.data = self.factor * s * self.pre * x
    def Height (self):
        return self.pre.height
    def Width (self):
        return self.pre.width
    def CreateVector(self):
        return self.matrix.CreateColVector()
    def CreateColVector(self):
        return self.matrix.CreateColVector()
    def CreateRowVector(self):
        return self.matrix.CreateRowVector()

class ZeroMatrix(BaseMatrix):
    def __init__ (self, width, height):
        super(ZeroMatrix, self).__init__()
        self.w = width
        self.h = height
    def MultAdd(self, s, x, y):
        pass
    def MultTransAdd(self, s, x, y):
        pass
    def Height (self):
        return self.h
    def Width (self):
        return self.w
    def CreateVector(self):
        return self.CreateColVector()
    def CreateColVector(self):
        return Vector(self.h)
    def CreateRowVector(self):
        return Vector(self.w)

class MatrixAB(BaseMatrix):
    def __init__ (self, a, b):
        super(MatrixAB, self).__init__()
        self.a = a
        self.b = b
    def MultAdd(self, s, x, y):
        y[0].data += s * self.a * x[0]
        y[1].data += s * self.b * x[0]
    def MultTransAdd(self, s, x, y):
        y[0].data += s * self.a * x[0] + s * self.b.T * x[1]
    def Height (self):
        return self.a.height + self.b.height
    def Width (self):
        return self.a.width + self.b.height
    def CreateVector(self):
        return self.CreateColVector()
    def CreateColVector(self):
        return Vector(self.height)
    def CreateRowVector(self):
        return Vector(self.width)

def BramblePasciakCG(matA, matB, matC, rhsUpper, rhsLower, preA, preM, sol=None, tol=1e-12, maxsteps=1000, printrates=True, initialize = True):
    k = 1 / min(EigenValues_Preconditioner(mat=matA, pre=preA))
    print("scale factor: ", k)
    scaledPreA = ScaledPreconditioner(k, matA, preA)

    matR = BlockMatrix([[matA, matB.T], [matB, matC]])
    matS = BlockMatrix([[scaledPreA, None], [None, IdentityMatrix(matB.height)]])
    matT = BlockMatrix([[IdentityMatrix(matA.width), None], [matB, -IdentityMatrix(matB.height)]])
    matAB = MatrixAB(matA, matB)
    matPTS = matAB @ matS - IdentityMatrix(matA.height + matB.height)
    matK = matPTS @ matR
    preSchurComplement = BlockMatrix([[IdentityMatrix(matA.width), None], [None, preM]])
    matPreTimesPTS = preSchurComplement @ matT @ matS
    matPreTimesK = matPreTimesPTS @ matR

    rhs = BlockVector([rhsUpper, rhsLower])
    u = sol if sol else rhs.CreateVector()
    if initialize: u[:] = 0.0
    d = rhs.CreateVector()
    p = rhs.CreateVector()
    pTemp = rhs.CreateVector()
    s = rhs.CreateVector()
    preValue = rhs.CreateVector()

    pTemp.data = rhs - matR * u
    s.data = matS * pTemp
    d.data = matAB * s - rhs + matR * u
    p.data = preSchurComplement @ matT * s
    pTemp.data = p

    pdn = InnerProduct(p, d)
    err0 = sqrt(abs(pdn))
    print("error 0: ", err0)

    if pdn == 0:
        return u

    for it in range(maxsteps):
        p.data = -matR * pTemp
        preValue.data = -matS * p
        p.data += matAB * preValue
        pd = pdn
        as_s = InnerProduct(pTemp, p)
        alpha = pd / as_s
        u.data += alpha * pTemp
        d.data += (-alpha) * p
        s.data += (-alpha) * preValue

        p.data = preSchurComplement @ matT * s

        pdn = InnerProduct(p, d)
        beta = pdn / pd

        pTemp *= beta
        pTemp.data += p

        err = sqrt(abs(pd))
        if printrates:
            print ("it = ", it, " err = ", err)
        if err < tol * err0: break
    else:
        print("Warning: CG did not converge to TOL")

    return u

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
mesh = Mesh( geo.GenerateMesh(maxh=0.09))

mesh.Curve(3)

V = VectorH1(mesh, order=3, dirichlet="wall|inlet|cyl")
Q = H1(mesh, order=2)

u,v = V.TnT()
p,q = Q.TnT()

a = BilinearForm(V)
a += SymbolicBFI(InnerProduct(grad(u),grad(v)))

preA = Preconditioner(a, 'local')

b = BilinearForm(trialspace=V, testspace=Q)
b += SymbolicBFI(div(u)*q)

a.Assemble()
b.Assemble()

mp = BilinearForm(Q)
mp += SymbolicBFI(p*q)
preM = Preconditioner(mp, 'local')
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
sol = BlockVector( [gfu.vec, gfp.vec] )

with TaskManager(pajetrace=100*1000*1000):
    bramblePasciakTimer = Timer("BramblePasciakCG")
    bramblePasciakTimer.Start()
    BramblePasciakCG(a.mat, b.mat, None, f.vec, g.vec, preA, preM, sol, initialize=False, tol=1e-7, maxsteps=1000)
    bramblePasciakTimer.Stop()

    gfu = GridFunction(V, name="u")
    gfp = GridFunction(Q, name="p")
    uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
    gfu.Set(uin, definedon=mesh.Boundaries("inlet"))
    sol = BlockVector( [gfu.vec, gfp.vec] )
    K = BlockMatrix( [ [a.mat, b.mat.T], [b.mat, None] ] )
    C = BlockMatrix( [ [preA, None], [None, preM] ] )
    rhs = BlockVector ( [f.vec, g.vec] )
    sol = BlockVector( [gfu.vec, gfp.vec] )
    minResTimer = Timer("MinRes")
    minResTimer.Start()
    solvers.MinRes(mat=K, pre=C, rhs=rhs, sol=sol, initialize=False, tol=1e-7, maxsteps=1000)
    minResTimer.Stop()

XV = VectorH1(mesh, order=2, dirichlet="wall|inlet|cyl")
V.SetOrder(TRIG,3)
V.Update()
Q = L2(mesh, order=1)
X = FESpace([V,Q])
u,p = X.TrialFunction()
v,q = X.TestFunction()
a = BilinearForm(X)
a += SymbolicBFI(InnerProduct(grad(u),grad(v))+div(u)*q+div(v)*p)
a.Assemble()
gfu = GridFunction(X)
uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
directTimer = Timer("direct")
directTimer.Start()
res = gfu.vec.CreateVector()
res.data = -a.mat * gfu.vec
inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse="umfpack")
gfu.vec.data += inv * res
directTimer.Stop()
Draw(Norm(gfu.components[0]), mesh, "vel")

print("BramblePasciakCG took", bramblePasciakTimer.time, "seconds")
print("MinRes took", minResTimer.time, "seconds")
print("direct solver took", directTimer.time, "seconds")
