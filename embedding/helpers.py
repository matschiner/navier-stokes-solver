from ngsolve import *

dS = dx(element_boundary=True)


class EmbeddingTransformation(BaseMatrix):
    def __init__(self, M, Mw, eblocks, fblocks):
        super(EmbeddingTransformation, self).__init__()
        self.smootherE = Mw.mat.CreateBlockSmoother(eblocks) if mpi_world.size==1 else  Mw.mat.local_mat.CreateBlockSmoother(eblocks)
        self.smootherF = Mw.mat.CreateBlockSmoother(fblocks) if mpi_world.size==1 else  Mw.mat.local_mat.CreateBlockSmoother(fblocks)
        self.M = M
        self.Mw = Mw
        self.Mw_trans = Mw.mat.CreateTranspose() if mpi_world.size==1 else Mw.mat.local_mat.CreateTranspose()
        # self.smootherReversed = Mw_trans.CreateBlockSmoother(fblocks + eblocks)

    def Mult(self, x, y):
        rhs = y.CreateVector()
        rhs.data = self.M.mat * x
        y.data = self.smootherE * rhs
        rhs.data -= self.Mw.mat * y
        y.data += self.smootherF * rhs

        # y.data[:] = 0
        # self.smoother.Smooth(y, tmp)

    def MultTrans(self, x, y):
        tmp1 = x.CreateVector()
        tmp2 = x.CreateVector()

        tmp1.data = self.smootherF.T * x
        tmp2.data = x - self.Mw_trans * tmp1
        tmp1.data += self.smootherE.T * tmp2

        y.data = self.M.mat.T * tmp1

        # tmp.data[:] = 0
        # self.smootherReversed.Smooth(tmp, x)
        # y.data = M.mat.T * tmp

    def CreateRowVector(self):
        return self.M.mat.CreateRowVector()

    def CreateColVector(self):
        return self.Mw.mat.CreateColVector()

    def Width(self):
        return self.Mw.mat.width

    def Height(self):
        return self.M.mat.height


def CreateEmbeddingPreconditioner(X, nu, condense=False):
    mesh = X.mesh
    (u, u_hat, sigma), (v, v_hat, tau) = X.TnT()

    n = specialcf.normal(mesh.dim)

    def tang(vec):
        return vec - (vec * n) * n

    v_dual = v.Operator("dual")

    VH1 = VectorH1(mesh, order=1, dirichlet="wall|inlet|cyl")
    vH1trial, vH1test = VH1.TnT()

    M = BilinearForm(trialspace=VH1, testspace=X)
    M += vH1trial * v_dual * dx
    M += vH1trial * v_dual * dS
    M += vH1trial * tang(v_hat) * dS
    M.Assemble()

    Mw = BilinearForm(X, eliminate_hidden=False)
    Mw += u * v_dual * dx
    Mw += u * v_dual * dS
    Mw += u_hat * tang(v_hat) * dS
    Mw.Assemble()



    # Mw_inverse = Mw.mat.Inverse(inverse="umfpack")
    # Mw_trans_inverse = Mw_trans.Inverse(inverse="umfpack")

    # E = Mw_inverse @ M.mat
    # ET = M.mat.T @ Mw_trans_inverse

    laplaceH1 = BilinearForm(VH1, condense=condense)
    laplaceH1 += nu * InnerProduct(grad(vH1trial), grad(vH1test)) * dx

    laplaceH1_inverse = Preconditioner(laplaceH1, "bddc")
    laplaceH1.Assemble()
    # laplaceH1_inverse = laplaceH1.mat.Inverse(freedofs=VH1.FreeDofs(condense), inverse="sparsecholesky")

    eblocks = []
    for f in mesh.facets:  # edges in 2d, faces in 3d
        eblocks.append(X.GetDofNrs(f))

    fblocks = []
    for f in mesh.faces:
        # remove hidden dofs (=-2)
        fblocks.append([d for d in X.GetDofNrs(f) if d != -2])

    emb = EmbeddingTransformation(M, Mw, eblocks, fblocks)

    # test if embedding transformation is doing the right thing
    # gfh1 = GridFunction(VH1)
    # gfx = GridFunction(X)
    # gfh1.Set((x, x * x))
    # gfx.vec.data = emb * gfh1.vec
    # Draw(gfx.components[0])
    # input("lj")

    return emb @ laplaceH1_inverse @ emb.T
