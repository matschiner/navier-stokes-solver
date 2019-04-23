"""implementation of MinRes method"""

from ngsolve.la import InnerProduct
from math import sqrt
from ngsolve import Projector, Norm
from ngsolve.ngstd import Timer
import ngsolve

# Source: Michael Kolmbauer https://www.numa.uni-linz.ac.at/Teaching/PhD/Finished/kolmbauer-diss.pdf


def MinRes(mat, rhs, pre=None, sol=None, maxsteps=100, printrates=True, initialize=True, tol=1e-7):
    """Minimal Residuum method


    Parameters
    ----------

    mat : Matrix
      The left hand side of the equation to solve

    rhs : Vector
      The right hand side of the equation.

    pre : Preconditioner
      If provided the preconditioner is used.

    sol : Vector
      Start vector for MinRes method, if initialize is set False. Gets overwritten by the solution vector. If sol = None then a new vector is created.

    maxsteps : int
      Number of maximal steps for MinRes. If the maximal number is reached before the tolerance is reached MinRes stops.

    printrates : bool
      If set to True then the error of the iterations is displayed.

    initialize : bool
      If set to True then the initial guess for the MinRes method is set to zero. Otherwise the values of the vector sol, if prevented, is used.

    tol : double
      Tolerance of the residuum. MinRes stops if tolerance is reached.


    Returns
    -------
    (vector)
      Solution vector of the MinRes method.

    """
    u = sol if sol else rhs.CreateVector()

    v_new = rhs.CreateVector()
    v = rhs.CreateVector()
    v_old = rhs.CreateVector()
    w_new = rhs.CreateVector()
    w = rhs.CreateVector()
    w_old = rhs.CreateVector()
    z_new = rhs.CreateVector()
    z = rhs.CreateVector()
    mz = rhs.CreateVector()

    if (initialize):
        u[:] = 0.0
        v.data = rhs
    else:
        v.data = rhs - mat * u

    z.data = pre * v if pre else v

    # First Step
    gamma = sqrt(InnerProduct(z, v))
    gamma_new = 0
    z.data = 1 / gamma * z
    v.data = 1 / gamma * v

    ResNorm = gamma
    ResNorm_old = gamma

    if (printrates):
        print("it = ", 0, " err = ", ResNorm)

    eta_old = gamma
    c_old = 1
    c = 1
    s_new = 0
    s = 0
    s_old = 0

    v_old[:] = 0.0
    w_old[:] = 0.0
    w[:] = 0.0

    k = 1
    errors = [gamma]
    while (k < maxsteps + 1 and ResNorm > tol):
        mz.data = mat * z
        delta = InnerProduct(mz, z)
        v_new.data = mz - delta * v - gamma * v_old

        z_new.data = pre * v_new if pre else v_new

        gamma_new = sqrt(InnerProduct(z_new, v_new))
        z_new *= 1 / gamma_new
        v_new *= 1 / gamma_new

        alpha0 = c * delta - c_old * s * gamma
        alpha1 = sqrt(alpha0 * alpha0 + gamma_new * gamma_new)  # **
        alpha2 = s * delta + c_old * c * gamma
        alpha3 = s_old * gamma

        c_new = alpha0 / alpha1
        s_new = gamma_new / alpha1

        w_new.data = z - alpha3 * w_old - alpha2 * w
        w_new.data = 1 / alpha1 * w_new

        u.data += c_new * eta_old * w_new
        eta = -s_new * eta_old

        # update of residuum
        ResNorm = abs(s_new) * ResNorm_old
        if (printrates):
            print("it = ", k, " err = ", ResNorm)
        errors.append(ResNorm)
        if ResNorm < tol:
            break
        k += 1

        # shift vectors by renaming
        v_old, v, v_new = v, v_new, v_old
        w_old, w, w_new = w, w_new, w_old
        z, z_new = z_new, z

        eta_old = eta

        s_old = s
        s = s_new

        c_old = c
        c = c_new

        gamma = gamma_new
        ResNorm_old = ResNorm
    else:
        print("Warning: MinRes did not converge to TOL")

    return (u, errors)
