from ngsolve import *
from netgen.geom2d import unit_square

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

fesh1 = VectorH1(mesh, order=2)
feshdiv = HDiv(mesh, order=2)
fesvec = TangentialFacetFESpace(mesh, order=2)
fescomp = FESpace([feshdiv, fesvec])

gfuh1 = GridFunction(fesh1)
gfuh1.Set((x * x, y * y))

amixed = BilinearForm(trialspace=fesh1, testspace=fescomp)
acomp = BilinearForm(fescomp)

(u, uf), (v, vf) = fescomp.TnT()
vdual = v.Operator("dual")
uh1 = fesh1.TrialFunction()

n = specialcf.normal(mesh.dim)


def tang(v): return v - (v * n) * n


vf = tang(vf)

dS = dx(element_boundary=True)
acomp += u * vdual * dx
acomp += u * vdual * dS
acomp += uf * vf * dS
acomp.Assemble()

amixed += uh1 * vdual * dx
amixed += uh1 * vdual * dS
amixed += uh1 * vf * dS
amixed.Assemble()

print("ahdiv =", acomp.mat)
print("amixed =", amixed.mat)

gfucomp = GridFunction(fescomp, name="ucomp")
# transform = ahdiv.mat.Inverse() @ amixed.mat
# gfuhdiv.vec.data = transform * gfuh1.vec

eblocks = []
for e in mesh.edges:
    eblocks.append(fescomp.GetDofNrs(e))
fblocks = []
for f in mesh.faces:
    fblocks.append(fescomp.GetDofNrs(f))

print(eblocks)
print(fblocks)

einv = acomp.mat.CreateBlockSmoother(eblocks)
finv = acomp.mat.CreateBlockSmoother(fblocks)

transform = (einv + finv @ (IdentityMatrix() - acomp.mat @ einv)) @ amixed.mat
gfucomp.vec.data = transform * gfuh1.vec

Draw(gfuh1)
Draw(gfucomp.components[0], mesh, "uhdiv")
