from ngsolve import *
from ngsolve.krylovspace import MinRes

from templates.mt_global import *
from solvers.bramblepasciak import BramblePasciakCG

__all__ = ["NavierStokes"]


class NavierStokes:

    def __init__(self, mesh, nu, inflow, outflow, wall, uin, timestep, order=2, volumeforce=None):

        self.nu = nu
        self.timestep = timestep
        self.uin = uin
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall

        V = HDiv(mesh, order=order, dirichlet=inflow + "|" + wall, RT=False)
        self.V = V
        Vhat = TangentialFacetFESpace(mesh, order=order - 1, dirichlet=inflow + "|" + wall + "|" + outflow)
        Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)
        if mesh.dim == 2:
            S = L2(mesh, order=order - 1)
        else:
            S = VectorL2(mesh, order=order - 1)

        Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
        Sigma = Compress(Sigma)
        S.SetCouplingType(IntRange(0, S.ndof), COUPLING_TYPE.HIDDEN_DOF)
        S = Compress(S)

        self.X = FESpace([V, Vhat, Sigma, S])
        for i in range(self.X.ndof):
            if self.X.CouplingType(i) == COUPLING_TYPE.WIREBASKET_DOF:
                self.X.SetCouplingType(i, COUPLING_TYPE.INTERFACE_DOF)
        self.v1dofs = self.X.Range(0)

        # for iterative method
        self.X2 = FESpace([V, Vhat, Sigma, S])
        for f in mesh.facets:
            self.X2.SetCouplingType(V.GetDofNrs(f)[1], COUPLING_TYPE.WIREBASKET_DOF)
            self.X2.SetCouplingType(V.ndof + Vhat.GetDofNrs(f)[1], COUPLING_TYPE.WIREBASKET_DOF)

        u, uhat, sigma, W = self.X.TrialFunction()
        v, vhat, tau, R = self.X.TestFunction()

        if mesh.dim == 2:
            def Skew2Vec(m):
                return m[1, 0] - m[0, 1]
        else:
            def Skew2Vec(m):
                return CoefficientFunction((m[0, 1] - m[1, 0], m[2, 0] - m[0, 2], m[1, 2] - m[2, 1]))

        dS = dx(element_boundary=True)
        n = specialcf.normal(mesh.dim)

        def tang(u):
            return u - (u * n) * n

        self.stokesA = -0.5 / nu * InnerProduct(sigma, tau) * dx + \
                       (div(sigma) * v + div(tau) * u) * dx + \
                       (InnerProduct(W, Skew2Vec(tau)) + InnerProduct(R, Skew2Vec(sigma))) * dx + \
                       -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
                       (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS

        self.V_trace = 2 * nu * div(u) * div(v) * dx

        self.astokes = BilinearForm(self.X, eliminate_hidden=True)
        self.astokes += self.stokesA
        self.astokes += 1e12 * nu * div(u) * div(v) * dx
        self.pre_astokes = Preconditioner(self.astokes, "bddc")

        self.a = BilinearForm(self.X, eliminate_hidden=True)
        self.a += self.stokesA

        self.gfu = GridFunction(self.X)
        self.f = LinearForm(self.X)

        self.mstar = BilinearForm(self.X, eliminate_hidden=True, condense=True)
        self.mstar += u * v * dx + timestep * self.stokesA

        self.premstar = Preconditioner(self.mstar, "bddc")
        self.mstar.Assemble()
        # self.invmstar = self.mstar.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")
        # self.invmstar1 = self.mstar.mat.Inverse(self.X.FreeDofs(self.mstar.condense), inverse="sparsecholesky")

        self.invmstar1 = CGSolver(self.mstar.mat, pre=self.premstar, precision=1e-4, printrates=False)
        ext = IdentityMatrix(self.X.ndof) + self.mstar.harmonic_extension
        extT = IdentityMatrix(self.X.ndof) + self.mstar.harmonic_extension_trans
        self.invmstar = ext @ self.invmstar1 @ extT + self.mstar.inner_solve

        if False:
            u, v = V.TnT()
            self.conv = BilinearForm(V, nonassemble=True)
            self.conv += SymbolicBFI(InnerProduct(grad(v) * u, u).Compile(True, wait=True))
            self.conv += SymbolicBFI((-IfPos(u * n, u * n * u * v, u * n * u.Other(bnd=self.uin) * v)).Compile(True, wait=True), element_boundary=True)
            emb = Embedding(self.X.ndof, self.v1dofs)
            self.conv_operator = emb @ self.conv.mat @ emb.T
        else:
            VL2 = VectorL2(mesh, order=order, piola=True)
            ul2, vl2 = VL2.TnT()
            self.conv_l2 = BilinearForm(VL2, nonassemble=True)
            self.conv_l2 += InnerProduct(grad(vl2) * ul2, ul2).Compile(realcompile=realcompile, wait=True) * dx
            self.conv_l2 += (-IfPos(ul2 * n, ul2 * n * ul2 * vl2, ul2 * n * ul2.Other(bnd=self.uin) * vl2)).Compile(realcompile=realcompile, wait=True) * dS

            self.convertl2 = V.ConvertL2Operator(VL2) @ Embedding(self.X.ndof, self.v1dofs).T
            self.conv_operator = self.convertl2.T @ self.conv_l2.mat @ self.convertl2

        self.V2 = HDiv(mesh, order=order, RT=False, discontinuous=True)
        self.Q = L2(mesh, order=order - 1)
        self.Qhat = FacetFESpace(mesh, order=order, dirichlet=outflow)
        self.Xproj = FESpace([self.V2, self.Q, self.Qhat])
        (u, p, phat), (v, q, qhat) = self.Xproj.TnT()
        aproj = BilinearForm(self.Xproj, condense=True)
        aproj += (-u * v + div(u) * q + div(v) * p) * dx + (u * n * qhat + v * n * phat) * dS
        cproj = Preconditioner(aproj, "bddc", coarsetype="h1amg")
        aproj.Assemble()

        # self.invproj1 = aproj.mat.Inverse(self.Xproj.FreeDofs(aproj.condense), inverse="sparsecholesky")
        self.invproj1 = CGSolver(aproj.mat, pre=cproj, printrates=False)
        ext = IdentityMatrix() + aproj.harmonic_extension
        extT = IdentityMatrix() + aproj.harmonic_extension_trans
        self.invproj = ext @ self.invproj1 @ extT + aproj.inner_solve

        self.bproj = BilinearForm(trialspace=self.V, testspace=self.Xproj)
        self.bproj += SymbolicBFI(div(self.V.TrialFunction()) * q)
        self.bproj.Assemble()

        # mapping of discontinuous to continuous H(div)
        ind = self.V.ndof * [0]
        for el in mesh.Elements(VOL):
            dofs1 = self.V.GetDofNrs(el)
            dofs2 = self.V2.GetDofNrs(el)
            for d1, d2 in zip(dofs1, dofs2):
                ind[d1] = d2
        self.mapV = PermutationMatrix(self.Xproj.ndof, ind)

    @property
    def velocity(self):
        return self.gfu.components[0]

    @property
    def pressure(self):
        return 1e6 / self.nu * div(self.gfu.components[0])

    @property
    def pressureExpl(self):
        return self.gfp

    def SolveInitial(self, timesteps=None, iterative=True):
        self.a.Assemble()
        self.f.Assemble()

        self.gfu.components[0].Set(self.uin, definedon=self.X.mesh.Boundaries(self.inflow))
        self.gfu.components[1].Set(self.uin, definedon=self.X.mesh.Boundaries(self.inflow))

        if not timesteps:
            if iterative:
                p, q = self.Q.TnT()
                u, uhat, sigma, W = self.X2.TrialFunction()
                # v, vhat, tau, R = self.X2.TestFunction()

                blfA = BilinearForm(self.X2, eliminate_hidden=True, condense=True)
                blfA += self.stokesA
                blfA += self.V_trace
                preA = Preconditioner(blfA, "bddc")
                blfA.Assemble()

                g = LinearForm(self.Q)
                g.Assemble()

                mp = BilinearForm(self.Q)
                mp += p * q * dx
                preM = Preconditioner(mp, 'local')
                mp.Assemble()

                blfB = BilinearForm(trialspace=self.X2, testspace=self.Q)
                blfB += div(u) * q * dx
                blfB.Assemble()

                self.gfp = GridFunction(self.Q)
                self.gfp.vec.data = g.vec
                sol = BlockVector([self.gfu.vec, self.gfp.vec])
                if blfA.condense:
                    self.f.vec.data += blfA.harmonic_extension_trans * self.f.vec
                BramblePasciakCG(blfA, blfB, None, self.f.vec, g.vec, preA, preM, sol, initialize=False, tol=1e-9, maxsteps=100000, rel_err=True)
                if blfA.condense:
                    sol[0].data += blfA.harmonic_extension * sol[0]
                    sol[0].data += blfA.inner_solve * self.f.vec
            else:
                self.astokes.Assemble()
                temp = self.a.mat.CreateColVector()
                temp.data = -self.astokes.mat * self.gfu.vec + self.f.vec
                inv = self.astokes.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")

                self.gfu.vec.data += inv * temp
        else:
            self.Project(self.gfu.vec[0:self.V.ndof])
            for it in range(timesteps):
                print("it =", it)
                self.temp = self.a.mat.CreateColVector()
                self.temp2 = self.a.mat.CreateColVector()
                # self.f.Assemble()
                # self.temp.data = self.conv_operator * self.gfu.vec
                # self.temp.data += self.f.vec
                self.temp.data = -self.a.mat * self.gfu.vec

                self.temp2.data = self.invmstar * self.temp
                self.Project(self.temp2[0:self.V.ndof])
                self.gfu.vec.data += self.timestep * self.temp2.data
                self.Project(self.gfu.vec[0:self.V.ndof])

    def AddForce(self, force):
        force = CoefficientFunction(force)
        v, vhat, tau, R = self.X.TestFunction()
        self.f += SymbolicLFI(force * v)

    def DoTimeStep(self):

        self.temp = self.a.mat.CreateColVector()
        self.temp2 = self.a.mat.CreateColVector()
        self.f.Assemble()
        self.temp.data = self.conv_operator * self.gfu.vec
        self.temp.data += self.f.vec
        self.temp.data += -self.a.mat * self.gfu.vec

        self.temp2.data = self.invmstar * self.temp
        self.Project(self.temp2[0:self.V.ndof])
        self.gfu.vec.data += self.timestep * self.temp2.data

    def Project(self, vel):
        vel.data -= (self.mapV @ self.invproj @ self.bproj.mat) * vel
