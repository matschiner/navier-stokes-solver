from ngsolve.la import *
from ngsolve.la import EigenValues_Preconditioner
from math import sqrt
from ngsolve import Projector, Norm
from ngsolve.ngstd import Timer
import ngsolve


class Matrix_CInv_K(BaseMatrix):
    def __init__(self, mother):
        super(Matrix_CInv_K, self).__init__()
        self.m = mother

    def MultAdd(self, s, l, r):
        r[0].data += s * self.m.tmp1
        r[1].data += s * self.m.M_inv * self.m.tmp3


class Matrix_K(BaseMatrix):
    def __init__(self, mother):
        super(Matrix_K, self).__init__()
        self.m = mother

    def MultAdd(self, s, l, r):
        r[0].data += s * (self.m.tmp2 - self.m.tmp0)
        r[1].data += s * self.m.tmp3


class BP_Matrices():
    def __init__(self, A0_inv, A, B, M_inv):
        self.A0_inv = A0_inv
        self.A = A
        self.B = B
        self.M_inv = M_inv

        self.tmpA = self.A.CreateColVector()
        self.krylA = A.CreateColVector()
        self.krylA[:] = 0

        self.tmpB = self.B.CreateRowVector()
        self.krylB = B.CreateRowVector()
        self.krylB[:] = 0

        self.tmp0 = A.CreateColVector()
        self.tmp1 = A0_inv.CreateColVector()
        self.tmp2 = A.CreateColVector()
        self.tmp3 = B.CreateColVector()
        self.tmp4 = self.tmp1.CreateVector()
        self.tmp5 = self.tmp1.CreateVector()
        self.Cinv_K = Matrix_CInv_K(self)
        self.K = Matrix_K(self)

        self.u = A.CreateRowVector()
        self.v = B.CreateColVector()

    def update(self, s):
        self.u.data = s[0]
        self.v.data = s[1]

        self.tmpA.data = self.A * self.u
        self.tmpB.data = self.B.T * self.v

        self.krylA += self.tmpA
        self.krylB += self.tmpB

        self.tmp0.data = self.tmpA + self.tmpB  # A u + B.T v

        self.tmp1.data = self.A0_inv * self.tmp0  # A0^-1 (A u + B.T v)
        self.tmp2.data = self.A * self.tmp1
        self.tmp4.data = self.tmp1 - self.u
        self.tmp3.data = self.B * self.tmp4
        self.tmp5.data = self.tmp1 - self.tmp0

    def K_Inner(self, kryl=False, wd=0):
        tmp = InnerProduct(self.krylA if kryl else self.tmpA, self.tmp1) - InnerProduct(self.u, self.tmp0) + \
              InnerProduct(self.krylB if kryl else self.tmpB, self.tmp4)

        return tmp

    def KCK_Inner(self, kryl=False):
        tmp = InnerProduct(self.krylA if kryl else self.tmpA, self.tmp1) + \
              InnerProduct(self.krylB if kryl else self.tmpB, self.tmp4)
        return tmp


def BramblePasciakCG(matA, matB, matC, f, g, preA_unscaled, preM, sol=None, tol=1e-6, maxsteps=100, printrates=True,
                     initialize=True, rel_err=True):
    """preconditioned conjugate gradient method


    Parameters
    ----------

    mat : Matrix
      The left hand side of the equation to solve. The matrix has to be spd or hermitsch.

    rhs : Vector
      The right hand side of the equation.

    pre : Preconditioner
      If provided the preconditioner is used.

    sol : Vector
      Start vector for CG method, if initialize is set False. Gets overwritten by the solution vector. If sol = None then a new vector is created.

    tol : double
      Tolerance of the residuum. CG stops if tolerance is reached.

    maxsteps : int
      Number of maximal steps for CG. If the maximal number is reached before the tolerance is reached CG stops.

    printrates : bool
      If set to True then the error of the iterations is displayed.

    initialize : bool
      If set to True then the initial guess for the CG method is set to zero. Otherwise the values of the vector sol, if provided, is used.

    conjugate : bool
      If set to True, then the complex inner product is used.


    Returns
    -------
    (vector)
      Solution vector of the CG method.

    """

    lams = EigenValues_Preconditioner(mat=matA, pre=preA_unscaled)
    print("min", min(lams), "max", max(lams))
    k = 1. / min(lams) + 1e-3
    print("scale factor", k)
    preA = k * preA_unscaled

    f_new = matA.CreateColVector()
    tmp0 = preA.CreateColVector()
    tmp0.data = preA * f
    f_new.data = matA * tmp0 - f

    g_new = matB.CreateColVector()
    g_new.data = matB * tmp0 - g

    rhs = BlockVector([f_new, g_new])

    timer = Timer("BPCG-Solver")
    timer.Start()
    u = sol if sol else rhs.CreateVector()
    if initialize:
        u[:] = 0.0

    d = rhs.CreateVector()
    w = rhs.CreateVector()
    v = rhs.CreateVector()
    s = rhs.CreateVector()

    MatOp = BP_Matrices(preA, matA, matB, preM)

    MatOp.update(u)
    d.data = rhs - MatOp.K * u
    pr = rhs.CreateVector()
    pr[0].data = preA * f

    tmp1 = matB.CreateColVector()
    tmp1.data = matB * pr[0] - g

    pr[1].data = preM * tmp1

    w.data = pr - MatOp.Cinv_K * u

    wdn = InnerProduct(w, d)

    err0 = sqrt(abs(wdn))
    print("err0", err0)
    s.data = w

    if wdn == 0:
        return u
    wdn_2 = 0
    for it in range(maxsteps):

        MatOp.update(s)

        v.data = MatOp.K * s
        wd = wdn
        # as_s = InnerProduct(s, v)
        alpha = wd / MatOp.K_Inner()

        u.data += alpha * s
        d.data += (-alpha) * v
        w.data = w + (-alpha) * MatOp.Cinv_K * s

        wdn = InnerProduct(w, d)
        wdn_2 += MatOp.KCK_Inner(w)
        print("wdn", wdn, "wdn2", wdn_2)
        beta = wdn / wd

        s *= beta
        s.data += w

        err = sqrt(abs(wd))
        if printrates:
            print("\rit = ", it, " err = ", err, " " * 20, end="")
        if err < tol * (err0 if rel_err else 1):
            break
    else:
        print("Warning: BPCG did not converge to TOL")

    timer.Stop()
    print("\n")
    return k
