from ngsolve import *
import ngs_amg

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


def CreateEmbeddingPreconditioner(X, nu, condense=False, diri=".*", hodivfree=False, slip=False, slip_boundary=[], auxiliary_precon="direct"):
    mesh = X.mesh

    (u, u_hat, _, _), (v, v_hat, _, _) = X.TnT()
    xfree = X.FreeDofs()

    # projector_lo = Projector(xfree, True)

    n = specialcf.normal(mesh.dim)

    def tang(vec):
        return vec - (vec * n) * n

    v_dual = v.Operator("dual")
    VH1 = VectorH1(mesh, order=1, dirichlet=diri)

    vH1trial, vH1test = VH1.TnT()
    vchar = FacetFESpace(mesh, order=0)
    gfchar = GridFunction(vchar, "char")
    gfchar.vec.data[:] = 1
    if slip:
        for e in mesh.Elements(BND):
            if e.mat in slip_boundary:
                gfchar.vec[vchar.GetDofNrs(e)[0]] = 0

    M = BilinearForm(trialspace=VH1, testspace=X)
    M += vH1trial * v_dual * dx
    M += vH1trial * v_dual * gfchar * dS
    M += vH1trial * tang(v_hat) * dS
    M.Assemble()

    Mw = BilinearForm(X)
    Mw += u * v_dual * dx
    Mw += u * v_dual * dS
    Mw += u_hat * tang(v_hat) * dS
    Mw.Assemble()

    # Mw_inverse = Mw.mat.Inverse(inverse="umfpack")
    # Mw_trans_inverse = Mw_trans.Inverse(inverse="umfpack")

    # E = Mw_inverse @ M.mat
    # ET = M.mat.T @ Mw_trans_inverse
    ir = IntegrationRule([[0], [1]], [0.5, 0.5])

    if auxiliary_precon == "direct":
        laplaceH1 = BilinearForm(VH1, condense=condense)
        laplaceH1 += nu * 0.25 * InnerProduct(grad(vH1trial) + grad(vH1trial).trans, grad(vH1test) + grad(vH1test).trans) * dx
        if slip:
            laplaceH1 += nu / specialcf.mesh_size * vH1trial * n * vH1test * n * ds("cyl|wall", intrules={SEGM: ir})
        laplaceH1_inverse = Preconditioner(laplaceH1, "direct")
        laplaceH1.Assemble()
    elif auxiliary_precon == "h1amg":
        pc_opts = {"ngs_amg_max_coarse_size": 10,
                   # "ngs_amg_log_level": "normal",
                   "ngs_amg_do_test": True,
                   "ngs_amg_rots": False,
                   "ngs_amg_dof_blocks": [1, 1],
                   "ngs_amg_print_log": True}

        laplaceH1 = BilinearForm(VH1, condense=condense)
        laplaceH1 += nu * 0.25 * InnerProduct(grad(vH1trial) + grad(vH1trial).trans, grad(vH1test) + grad(vH1test).trans) * dx
        if slip:
            laplaceH1 += nu / specialcf.mesh_size * vH1trial * n * vH1test * n * ds("cyl|wall", intrules={SEGM: ir})
        laplaceH1_inverse = Preconditioner(laplaceH1, "ngs_amg.elast2d", **pc_opts)
        laplaceH1.Assemble()
    elif auxiliary_precon == "h1amg_componentwise":

        if mesh.dim == 2:
            VH1_1 = H1(mesh, order=1, dirichlet=diri)
            VH1_2 = H1(mesh, order=1, dirichlet=diri)
            vH1trial_1, vH1test_1 = VH1_1.TnT()
            vH1trial_2, vH1test_2 = VH1_2.TnT()

            laplaceH1_1 = BilinearForm(VH1_1, condense=condense)
            laplaceH1_1 += nu * InnerProduct(grad(vH1trial_1), grad(vH1test_1)) * dx

            laplaceH1_2 = BilinearForm(VH1_2, condense=condense)
            laplaceH1_2 += nu * InnerProduct(grad(vH1trial_2), grad(vH1test_2)) * dx

            pc_opts = {"ngs_amg_max_coarse_size": 5,
                       # "ngs_amg_log_level": "normal",
                       "ngs_amg_do_test": True,
                       "ngs_amg_print_log": True}
            laplaceH1_inverse_1 = Preconditioner(laplaceH1_1, "ngs_amg.h1_scal", **pc_opts)
            laplaceH1_inverse_2 = Preconditioner(laplaceH1_2, "ngs_amg.h1_scal", **pc_opts)
            laplaceH1_1.Assemble()
            laplaceH1_2.Assemble()

            emb_comp1 = Embedding(VH1.ndof, VH1.Range(0))
            emb_comp2 = Embedding(VH1.ndof, VH1.Range(1))
            laplaceH1_inverse = emb_comp1 @ laplaceH1_inverse_1 @ emb_comp1.T + emb_comp2 @ laplaceH1_inverse_2 @ emb_comp2.T

        else:
            VH1_1 = H1(mesh, order=1, dirichlet=diri)
            VH1_2 = H1(mesh, order=1, dirichlet=diri)
            VH1_3 = H1(mesh, order=1, dirichlet=diri)
            vH1trial_1, vH1test_1 = VH1_1.TnT()
            vH1trial_2, vH1test_2 = VH1_2.TnT()
            vH1trial_3, vH1test_3 = VH1_3.TnT()

            laplaceH1_1 = BilinearForm(VH1_1, condense=condense)
            laplaceH1_2 = BilinearForm(VH1_2, condense=condense)
            laplaceH1_3 = BilinearForm(VH1_3, condense=condense)

            laplaceH1_1 += nu * InnerProduct(grad(vH1trial_1), grad(vH1test_1)) * dx
            laplaceH1_2 += nu * InnerProduct(grad(vH1trial_2), grad(vH1test_2)) * dx
            laplaceH1_3 += nu * InnerProduct(grad(vH1trial_3), grad(vH1test_3)) * dx

            pc_opts = {"ngs_amg_max_coarse_size": 5,
                       # "ngs_amg_log_level": "normal",
                       "ngs_amg_do_test": True,
                       "ngs_amg_print_log": True}
            laplaceH1_inverse_1 = Preconditioner(laplaceH1_1, "ngs_amg.h1_scal", **pc_opts)
            laplaceH1_inverse_2 = Preconditioner(laplaceH1_2, "ngs_amg.h1_scal", **pc_opts)
            laplaceH1_inverse_3 = Preconditioner(laplaceH1_3, "ngs_amg.h1_scal", **pc_opts)
            laplaceH1_1.Assemble()
            laplaceH1_2.Assemble()
            laplaceH1_3.Assemble()

            emb_comp1 = Embedding(VH1.ndof, VH1.Range(0))
            emb_comp2 = Embedding(VH1.ndof, VH1.Range(1))
            emb_comp3 = Embedding(VH1.ndof, VH1.Range(1))
            laplaceH1_inverse = emb_comp1 @ laplaceH1_inverse_1 @ emb_comp1.T + emb_comp2 @ laplaceH1_inverse_2 @ emb_comp2.T + emb_comp3 @ laplaceH1_inverse_3 @ emb_comp3.T

    # laplaceH1_inverse=Preconditioner(laplaceH1)

    eblocks = []
    for f in mesh.facets:  # edges in 2d, faces in 3d
        eblocks.append(X.GetDofNrs(f))

    fblocks = []
    for f in mesh.faces:
        # remove hidden dofs (=-2)
        fblocks.append([d for d in X.GetDofNrs(f) if d != -2])

    emb = EmbeddingTransformation(M, Mw, eblocks, fblocks)

    # laplaceH1_inverse_wrapped = WrapMatrix(laplaceH1_inverse)
    proj = Projector(X.FreeDofs(True), True)
    emb = proj @ emb
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


"""   class WrapMatrix(BaseMatrix):
       def __init__(self, toWrap):
           super(WrapMatrix, self).__init__()
           self.toWrap = toWrap
           self.gfu = GridFunction(VH1, name="h1_sol")
           Draw(self.gfu)

       def Mult(self, x, y, ):
           print(x)
           self.gfu.vec.data = self.toWrap * x
           Redraw()
           input("continue")
           y.data = self.gfu.vec.data

       def CreateColVector(self):
           return self.toWrap.CreateColVector()

       def CreateRowVector(self):
           return self.toWrap.CreateRowVector()"""
