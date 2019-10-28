from ngsolve.la import *
from ngsolve.la import EigenValues_Preconditioner
from math import sqrt
from ngsolve import Projector, Norm
from ngsolve.ngstd import Timer


def harmonic_extension(f, blfA, inverse, result=None):
    if result is None:
        result = inverse.CreateColVector()
    if blfA.condense:
        f_residual = f.Copy()
        f_residual.data += blfA.harmonic_extension_trans * f_residual

        result.data = inverse * f_residual
        result.data += blfA.harmonic_extension * result
        result.data += blfA.inner_solve * f_residual
        return result
    else:
        result.data = inverse * f
        return result


def BramblePasciakCG(blfA, blfB, matC, f, g, preA_unscaled, preM, sol=None, tol=1e-6, maxsteps=100, printrates=True,
                     initialize=True, rel_err=True):
    """preconditioned bramble pasciak conjugate gradient method


    Parameters
    ----------

    matA : Matrix
      The left upper matrix of the saddle point matrix. The matrix has to be spd.

    matB : Matrix
      The B of the saddle point matrix. The matrix has to be spd.

    matC : Matrix
      The lower right of the saddle point matrix. The matrix has to be spd.

    f : Vector
      The first component right hand side of the equation.

    g : Vector
      The second component right hand side of the equation.

    preA_unscaled :
      Preconditioner of matA, preferedly BDDC.

    preA_unscaled :
      Preconditioner of matB, arbitrary (i.e. Jacobi).

    sol : Vector
      Start vector for CG method, if initialize is set False. Gets overwritten by the solution vector. If sol = None then a new vector is created.

    tol : double
      Tolerance of the residuum. BPCG stops if tolerance is reached.

    maxsteps : int
      Number of maximal steps for BPCG. If the maximal number is reached before the tolerance is reached CG stops.

    printrates : bool
      If set to True then the error of the iterations is displayed.

    initialize : bool
      If set to True then the initial guess for the BPCG method is set to zero. Otherwise the values of the vector sol, if provided, is used.

    conjugate : bool
      If set to True, then the complex inner product is used.

    rel_err : bool
      Whether to use the tolerance relative to the the initial error or not.

    Returns
    -------
    (vector - call by reference)
      Solution vector of the BPCG method.

    (int)
      Number of iterations the BCGP took

    """

    class myAmatrix(BaseMatrix):
        def __init__(self, blfA):
            super(myAmatrix, self).__init__()
            self.blfA = blfA
            self.mat = (IdentityMatrix() - blfA.harmonic_extension_trans) @ (blfA.mat + blfA.inner_matrix) @ (IdentityMatrix() - blfA.harmonic_extension)

        def Mult(self, x, y):
            y.data = self.mat * x

        def Height(self):
            return self.blfA.mat.height

        def Width(self):
            return self.blfA.mat.width

        def CreateColVector(self):
            return self.blfA.mat.CreateColVector()

        def CreateVector(self):
            return self.blfA.mat.CreateColVector()

    if blfA.condense:
        matA = myAmatrix(blfA)
    else:
        matA = blfA.mat
    matB = blfB.mat

    timer_prep = Timer("BPCG-Preparation")
    timer_prep.Start()
    timer_prepev = Timer("BPCG-Preparation-EV")
    timer_prepev.Start()
    lams = EigenValues_Preconditioner(mat=matA, pre=preA_unscaled, tol=1e-3)
    timer_prepev.Stop()
    # print("min", min(lams), "max", max(lams))
    k = 1. / min(lams) + 1e-3

    print("condition", max(lams) / min(lams))
    # print("scale factor", k)
    preA = k * preA_unscaled

    f_new = matA.CreateColVector()

    # tmp0.data = preA * f
    #tmp0 = harmonic_extension(f, blfA, preA)
    tmp0 = f.CreateVector()
    harmonic_extension(f, blfA, preA, result=tmp0)
    f_new.data = matA * tmp0 - f

    g_new = matB.CreateColVector()
    g_new.data = matB * tmp0 - g

    rhs = BlockVector([f_new, g_new])

    u = sol if sol else rhs.CreateVector()
    if initialize:
        u[:] = 0.0

    d = rhs.CreateVector()
    w = rhs.CreateVector()
    v = rhs.CreateVector()
    z = rhs.CreateVector()
    z_old = rhs.CreateVector()
    s = rhs.CreateVector()

    # MatOp = BP_Matrices(preA, matA, matB, preM)

    # MatOp.update(u)
    tmp0 = blfA.mat.CreateColVector()
    #tmp1 = preA.CreateColVector()
    tmp1 = blfA.mat.CreateColVector()
    tmp2 = blfA.mat.CreateColVector()
    tmp3 = matB.CreateColVector()
    tmp4 = tmp1.CreateVector()
    matA_s0 = blfA.mat.CreateColVector()
    matB_s1 = matB.CreateRowVector()

    tmp0.data = matA * u[0] + matB.T * u[1]
    # tmp1.data = preA * tmp0
    harmonic_extension(tmp0, blfA, preA, tmp1)
    tmp2.data = matA * tmp1

    tmp4.data = tmp1 - u[0]
    tmp3.data = matB * tmp4

    # d.data = rhs - MatOp.K * u
    d[0].data = rhs[0] - (tmp2 - tmp0)
    d[1].data = rhs[1] - tmp3

    pr = rhs.CreateVector()
    harmonic_extension(f, blfA, preA, pr[0])
    # pr[0].data = preA * f

    tmp5 = matB.CreateColVector()
    tmp5.data = matB * pr[0] - g

    pr[1].data = preM * tmp5

    # w.data = pr - MatOp.Cinv_K * u
    w[0].data = pr[0] - tmp1
    w[1].data = pr[1] - preM * tmp3

    wdn = InnerProduct(w, d)

    err0 = sqrt(abs(wdn))
    print("err0", err0)
    s.data = w

    if wdn == 0:
        return u

    timer_prep.Stop()
    timer_its = Timer("BPCG-Iterations")
    timer_its.Start()

    matB_tranposed = matB.CreateTranspose()

    for it in range(maxsteps):
        if it == 0:
            matA_s0.data = matA * s[0]
            z[0].data = matA_s0  # A*w_0_0
        else:
            matA_s0.data = beta * matA_s0 + z_old[0] - alpha * tmp2
        matB_s1.data = matB_tranposed * s[1]
        tmp0.data = matA_s0 + matB_s1
        # tmp1.data = preA * tmp0
        harmonic_extension(f=tmp0, blfA=blfA, inverse=preA, result=tmp1)
        tmp2.data = matA * tmp1

        tmp4.data = tmp1 - s[0]
        tmp3.data = matB * tmp4

        z_old[0].data = z[0]

        # w.data = MatOp.K * s
        v[0].data = tmp2 - tmp0
        v[1].data = tmp3

        wd = wdn
        as_s = InnerProduct(s, v)
        # giving one (A,B) to the other side of the dot product
        # as_s = InnerProduct(matA_s0, tmp1) - InnerProduct(s[0], tmp0) + InnerProduct(matB_s1, tmp4)

        alpha = wd / as_s

        u.data += alpha * s
        d.data += (-alpha) * v

        # w.data = w_old + (-alpha) * MatOp.Cinv_K * s
        w[0].data = w[0] + (-alpha) * tmp1
        w[1].data = w[1] + (-alpha) * preM * tmp3

        wdn = InnerProduct(w, d)
        beta = wdn / wd

        z[0].data -= alpha * tmp2

        s *= beta
        s.data += w

        err = sqrt(abs(wd))
        if printrates:
            print("it = ", it, " err = ", err, " " * 20)
        if err < tol * (err0 if rel_err else 1):
            break
    else:
        print("Warning: BPCG did not converge to TOL")

    timer_its.Stop()
    print("\n")
    return (it, timer_its.time)
