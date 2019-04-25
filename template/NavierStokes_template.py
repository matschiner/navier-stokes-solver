from ngsolve import *

# ngsglobals.msg_level=10

    
class NavierStokes:
    
    def __init__(self, mesh, nu, inflow, outflow, wall, uin, timestep, order=2, volumeforce=None):

        self.nu = nu
        self.timestep = timestep
        self.uin = uin
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall
        
        V1 = HDiv(mesh, order=order, dirichlet=inflow+"|"+wall, RT=False)
        Vhat = VectorFacet(mesh, order=order-1, dirichlet=inflow+"|"+wall+"|"+outflow)
        Sigma = HCurlDiv(mesh, order = order-1, orderinner=order, discontinuous=True)
        if mesh.dim == 2:
            S = L2(mesh, order=order-1)            
        else:
            S = VectorL2(mesh, order=order-1)
        self.V = FESpace ([V1,Vhat, Sigma, S])

        self.V.SetCouplingType(self.V.Range(2), COUPLING_TYPE.HIDDEN_DOF)
        self.V.SetCouplingType(self.V.Range(3), COUPLING_TYPE.HIDDEN_DOF)
        
        u, uhat, sigma, W  = self.V.TrialFunction()
        v, vhat, tau, R  = self.V.TestFunction()

        if mesh.dim == 2:
            def Vec2Skew(v):
                return CoefficientFunction((0, v, -v, 0), dims = (2,2))
        else:
            def Vec2Skew(v):
                return CoefficientFunction((0, v[0], -v[1], -v[0],  0,  v[2], v[1],  -v[2], 0), dims = (3,3))

        W = Vec2Skew(W)
        R = Vec2Skew(R)

        n = specialcf.normal(mesh.dim)
        def tang(u): return u-(u*n)*n
    
        self.a = BilinearForm (self.V, eliminate_hidden = True)
        self.a += SymbolicBFI ( 1/nu * InnerProduct ( sigma,tau))
        self.a += SymbolicBFI ( (div(sigma) * v + div(tau) * u))
        self.a += SymbolicBFI ( -(((sigma*n)*n ) * (v*n) + ((tau*n)*n )* (u*n)) , element_boundary = True)
        self.a += SymbolicBFI ( (sigma*n)*tang(vhat) + (tau*n)*tang(uhat), element_boundary = True)
        self.a += SymbolicBFI ( InnerProduct(W,tau) + InnerProduct(R,sigma) )
        self.a += SymbolicBFI ( 1e6/nu*div(u)*div(v))

        self.gfu = GridFunction(self.V)

        # self.f = LinearForm(self.V)

        self.mstar = BilinearForm(self.V, eliminate_hidden = True)
        self.mstar += SymbolicBFI ( timestep*1/nu * InnerProduct ( sigma,tau))
        self.mstar += SymbolicBFI ( timestep*(div(sigma) * v + div(tau) * u))
        self.mstar += SymbolicBFI ( timestep*(-(((sigma*n)*n ) * (v*n) + ((tau*n)*n )* (u*n))) , element_boundary = True)
        self.mstar += SymbolicBFI ( timestep*((sigma*n)*tang(vhat) + (tau*n)*tang(uhat)), element_boundary = True)
        self.mstar += SymbolicBFI ( timestep*(InnerProduct(W,tau) + InnerProduct(R,sigma)) )
        self.mstar += SymbolicBFI ( timestep*1e6/nu*div(u)*div(v))
        self.mstar += SymbolicBFI ( -u*v )


        self.conv = BilinearForm(self.V, nonassemble=True)
        # self.conv += SymbolicBFI(-InnerProduct(grad(u).trans*u, v).Compile(True, wait=True))
        self.conv += SymbolicBFI(InnerProduct(grad(v)*u, u).Compile(True, wait=True))
        self.conv += SymbolicBFI((-IfPos(u * n, u*n*u*v, u*n*u.Other(bnd=self.uin)*v)).Compile(True, wait=True), element_boundary = True)

        self.invmstar = None

    @property
    def Velocity(self):
        return self.Velocity()

    def Velocity(self):
        return self.gfu.components[0]
    @property
    def Pressure(self):
        return self.Pressure()
    def Pressure(self):
        return 1e6/self.nu*div(self.gfu.components[0])
        
    def SolveInitial(self):
        self.a.Assemble()

        temp = self.a.mat.CreateColVector()
        self.gfu.components[0].Set (self.uin, definedon=self.V.mesh.Boundaries(self.inflow))
        self.gfu.components[1].Set (self.uin, definedon=self.V.mesh.Boundaries(self.inflow))

        inv = self.a.mat.Inverse(self.V.FreeDofs(), inverse="sparsecholesky")
        temp.data = self.a.mat * self.gfu.vec
        self.gfu.vec.data -= inv * temp
        # solvers.BVP(bf=self.a, lf=self.f, gf=self.gfu, pre = None, maxsteps=10, tol = 1e-7)

        
    def DoTimeStep(self):
        if not self.invmstar:
            self.mstar.Assemble()
            self.invmstar = self.mstar.mat.Inverse(self.V.FreeDofs(), inverse="sparsecholesky")
            print ("nze(inv) = ", self.invmstar.nze)
            
        
        self.temp = self.a.mat.CreateColVector()        
        self.conv.Apply (self.gfu.vec, self.temp)
        self.temp.data += self.a.mat*self.gfu.vec
        self.gfu.vec.data -= self.timestep * self.invmstar * self.temp
        
        