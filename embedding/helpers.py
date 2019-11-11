from ngsolve import *

dS = dx(element_boundary=True)


class EmbeddingTransformation(BaseMatrix):
    def __init__(self, M, Mw, eblocks, fblocks):
        super(EmbeddingTransformation, self).__init__()
        par = mpi_world.size > 1
        get_loc = lambda X: X.local_mat if par else X
        self.M = get_loc(M.mat)
        self.Mw = get_loc(Mw.mat)
        self.smootherE = self.Mw.CreateBlockSmoother(eblocks)
        self.smootherF = self.Mw.CreateBlockSmoother(fblocks)
        self.Mw_trans = self.Mw.CreateTranspose()
        self.Vrhs = self.M.CreateColVector()  # embedding U -> V
        self.Vup = self.M.CreateColVector()
        # self.smootherReversed = Mw_trans.CreateBlockSmoother(fblocks + eblocks)

    def MultAdd(self, c, x, y):
        self.Vrhs.data = c * self.M * x

        # y.data = self.smootherE * rhs
        # rhs.data -= self.Mw * y
        # y.data += self.smootherF * rhs

        self.Vup.data = self.smootherE * self.Vrhs
        self.Vrhs -= self.Mw * self.Vup
        y.data += self.Vup
        y.data += self.smootherF * self.Vrhs

        # y.data[:] = 0
        # self.smoother.Smooth(y, tmp)

    def MultTransAdd(self, c, x, y):
        tmp1 = self.Vup
        tmp2 = self.Vrhs

        tmp1.data = self.smootherF.T * x
        tmp2.data = x - self.Mw_trans * tmp1
        tmp1.data += self.smootherE.T * tmp2

        y.data += c * self.M.T * tmp1

    def IsComplex(self):
        return False

    def CreateRowVector(self):
        return self.M.CreateRowVector()

    def CreateColVector(self):
        return self.M.CreateColVector()

    def Width(self):
        return self.M.width

    def Height(self):
        return self.M.height


def CreateEmbeddingPreconditioner(X, nu, condense=False, diri=".*"):
    mesh = X.mesh
    uu, vv = X.TnT()
    u, u_hat = uu[0], uu[1]
    v, v_hat = vv[0], vv[1]
    n = specialcf.normal(mesh.dim)

    def tang(vec):
        return vec - (vec * n) * n

    v_dual = v.Operator("dual")

    VH1 = VectorH1(mesh, order=1, dirichlet=diri)
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

    if mpi_world.size > 1:
        emb = ParallelMatrix(emb,
                             row_pardofs=M.mat.row_pardofs,
                             col_pardofs=M.mat.col_pardofs,
                             op=ParallelMatrix.C2C)

    # test if embedding transformation is doing the right thing
    # gfh1 = GridFunction(VH1)
    # gfx = GridFunction(X)
    # gfh1.Set((x, x * x))
    # gfx.vec.data = emb * gfh1.vec
    # Draw(gfx.components[0])
    # input("lj")

    return emb @ laplaceH1_inverse @ emb.T
