from netgen.geom2d import unit_square

from solvers import krylovspace as mysolvers
from multiplicative_precond.preconditioners import *
mesh = Mesh(unit_square.GenerateMesh(maxh=0.01))

fes = H1(mesh, order=3, dirichlet="left|bottom")
u, v = fes.TnT()
a = BilinearForm(fes)
a += SymbolicBFI(grad(u) * grad(v) + u * v)
a.Assemble()
f = LinearForm(fes)
f += SymbolicLFI(1 * v)
f.Assemble()
gfu = GridFunction(fes)
gfu_test = GridFunction(fes)

preJpoint = a.mat.CreateSmoother(fes.FreeDofs())

print("multiplicative Precond")

vertexdofs = BitArray(fes.ndof)
vertexdofs[:] = False

for v in mesh.vertices:
    for d in fes.GetDofNrs(v):
        vertexdofs[d] = True

vertexdofs &= fes.FreeDofs()

coarsepre = a.mat.Inverse(vertexdofs)

preGS = MultiplicativePrecond(preJpoint, coarsepre, a.mat)
mysolvers.CG(mat=a.mat, pre=preGS, rhs=f.vec, sol=gfu_test.vec, maxsteps=1000)

print("jacobi")

mysolvers.CG(mat=a.mat, pre=preJpoint, rhs=f.vec, sol=gfu.vec, maxsteps=1000)

gfu_test.vec.data -= gfu.vec
print("Difference", Norm(gfu_test.vec))

print("symmetric GS")
preGS = SymmetricGS(preJpoint)
mysolvers.CG(mat=a.mat, pre=preGS, rhs=f.vec, sol=gfu.vec, maxsteps=1000)

print("Block Jacobi")
blocks = []
freedofs = fes.FreeDofs()
for v in mesh.vertices:
    vdofs = set()
    for el in mesh[v].elements:
        vdofs |= set(d for d in fes.GetDofNrs(el) if freedofs[d])
    blocks.append(vdofs)

blockjac = a.mat.CreateBlockSmoother(blocks)

mysolvers.CG(mat=a.mat, pre=blockjac, rhs=f.vec, sol=gfu.vec, maxsteps=1000)
