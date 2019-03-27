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
    def Mult (self, x, y):
        y.data = self.pre * x
        y *= self.factor
    def MultTrans(self, x, y):
        y.data = self.pre.MultTrans(x, y)
        y *= self.factor
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

def BramblePasciakCG(matA, matB, matC, rhsUpper, rhsLower, preA=None, preM=None, sol=None, tol=1e-12, maxsteps=100, printrates=True, initialize = True):
    k = 1 / min(EigenValues_Preconditioner(mat=matA, pre=preA))
    print("scale factor: ", k)
    scaledPreA = ScaledPreconditioner(k, matA, preA)

    aTimesAPreMinusIdentity = matA @ scaledPreA - IdentityMatrix(matA.height)
    matT = BlockMatrix([[aTimesAPreMinusIdentity, None], [None, IdentityMatrix(matB.height)]])
    matN = BlockMatrix([[matA, matB.T], [matB @ aTimesAPreMinusIdentity.T, matB @ scaledPreA @ matB.T]])
    matM = matT @ matN
    preCTimesT = BlockMatrix([[scaledPreA, None], [None, preM]])

    timer = Timer("CG-Solver")
    timer.Start()
    tempUpper = rhsUpper.CreateVector()
    tempLower = rhsLower.CreateVector()
    tempUpper.data = preA * rhsUpper
    tempLower.data = matB * tempUpper - rhsLower
    tempUpper.data = aTimesAPreMinusIdentity * rhsUpper
    rhs = BlockVector([tempUpper, tempLower])

    u = sol if sol else rhs.CreateVector()
    d = rhs.CreateVector()
    p = rhs.CreateVector()
    pTemp = rhs.CreateVector()

    if initialize: u[:] = 0.0

    d.data = rhs - matM * u
    sUpper = rhsUpper.CreateVector()
    sLower = tempLower.CreateVector()
    sUpper.data = rhsUpper
    sLower.data = tempLower
    s = BlockVector([sUpper, sLower])
    s.data += -matN * u
    p.data = preCTimesT * s
    pTemp.data = p

    pdn = InnerProduct(p, d)
    err0 = sqrt(abs(pdn))
    print("error 0: ", err0)

    if pdn==0:
        return u

    for it in range(maxsteps):
        p.data = matM * pTemp
        pd = pdn
        as_s = InnerProduct(pTemp, p)
        alpha = pd / as_s
        u.data += alpha * pTemp
        s.data += (-alpha) * matN * pTemp
        d.data += (-alpha) * p

        p.data = preCTimesT * s

        pdn = InnerProduct(p, d)
        beta = pdn / pd

        pTemp *= beta
        pTemp.data += p

        err = sqrt(abs(pd))
        if printrates:
            print ("it = ", it, " err = ", err)
        if err < tol*err0: break
    else:
        print("Warning: CG did not converge to TOL")

    timer.Stop()
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
mesh = Mesh( geo.GenerateMesh(maxh=0.08))

mesh.Curve(3)

V = VectorH1(mesh, order=3, dirichlet="wall|inlet|cyl")
Q = H1(mesh, order=2)

u,v = V.TnT()
p,q = Q.TnT()

a = BilinearForm(V)
a += SymbolicBFI(InnerProduct(grad(u),grad(v)))

preA = Preconditioner(a, 'multigrid')

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

BramblePasciakCG(a.mat, b.mat, None, f.vec, g.vec, preA, preM, sol, initialize=False)

Draw(Norm(gfu.components[0]), mesh, "vel")

IdentityMatrix(10) @ None
