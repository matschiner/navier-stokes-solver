"""Implementation of Bramble Pasciak CG method"""

from math import sqrt
from ngsolve import *
from ngsolve.ngstd import Timer
from ngsolve.la import EigenValues_Preconditioner


class ScaledPreconditioner(BaseMatrix):
    def __init__(self, factor, matrix, pre):
        super(ScaledPreconditioner, self).__init__()
        self.factor = factor
        self.matrix = matrix
        self.pre = pre

    def MultAdd(self, s, x, y):
        y.data = self.factor * s * self.pre * x

    def MultTransAdd(self, s, x, y):
        # pre is assumed to be symmetric therefore leave out .T
        y.data = self.factor * s * self.pre * x

    def Height(self):
        return self.pre.height

    def Width(self):
        return self.pre.width

    def CreateVector(self):
        return self.matrix.CreateColVector()

    def CreateColVector(self):
        return self.matrix.CreateColVector()

    def CreateRowVector(self):
        return self.matrix.CreateRowVector()


class MatrixAB(BaseMatrix):
    def __init__(self, a, b):
        super(MatrixAB, self).__init__()
        self.a = a
        self.b = b

    def MultAdd(self, s, x, y):
        y[0].data += s * self.a * x[0]
        y[1].data += s * self.b * x[0]

    def Height(self):
        return self.a.height + self.b.height

    def Width(self):
        return self.a.width + self.b.height

    def CreateVector(self):
        return self.CreateColVector()

    def CreateColVector(self):
        return Vector(self.height)

    def CreateRowVector(self):
        return Vector(self.width)


def bramble_pasciak_cg(a_matrix, b_matrix, c_matrix, pre_a, pre_schur_complement,
                       upper_rhs, lower_rhs, solution=None,
                       tolerance=1e-12, max_steps=1000, print_rates=True):
    k = 1 / min(EigenValues_Preconditioner(mat=a_matrix, pre=pre_a)) + 1e-3
    print("scale factor: ", k)
    scaled_pre_a = ScaledPreconditioner(k, a_matrix, pre_a)

    original_matrix = BlockMatrix([[a_matrix, b_matrix.T],
                                   [b_matrix, c_matrix]])
    full_pre_a = BlockMatrix([[scaled_pre_a, None],
                              [None, IdentityMatrix(b_matrix.height)]])
    full_b = BlockMatrix([[IdentityMatrix(a_matrix.width), None],
                          [b_matrix, -IdentityMatrix(b_matrix.height)]])
    a_b_block_matrix = MatrixAB(a_matrix, b_matrix)
    full_pre_schur_complement = BlockMatrix([[IdentityMatrix(a_matrix.width), None],
                                             [None, pre_schur_complement]])

    rhs = BlockVector([upper_rhs, lower_rhs])
    if not solution:
        solution = rhs.CreateVector()
        solution[:] = 0

    residuum = rhs.CreateVector()
    temp_1 = rhs.CreateVector()
    full_preconditioned_residuum = rhs.CreateVector()
    a_preconditioned_residuum = rhs.CreateVector()
    temp_2 = rhs.CreateVector()

    temp_2.data = rhs - original_matrix * solution
    a_preconditioned_residuum.data = full_pre_a * temp_2
    residuum.data = a_b_block_matrix * a_preconditioned_residuum \
        - rhs + original_matrix * solution
    temp_1.data = full_pre_schur_complement @ full_b * a_preconditioned_residuum
    full_preconditioned_residuum.data = temp_1

    current_residual_error_squared = InnerProduct(temp_1, residuum)
    errors = []

    for iteration in range(max_steps):
        iteration_timer = Timer(
            "Bramble Pasciak CG Iteration " + str(iteration))
        iteration_timer.Start()

        err = sqrt(abs(current_residual_error_squared))
        if print_rates:
            print("\rit = ", iteration, " err = ", err, " " * 20, end="")
        errors.append(err)
        if err < tolerance:
            iteration_timer.Stop()
            break

        previous_residual_error_squared = current_residual_error_squared

        temp_1.data = -original_matrix * full_preconditioned_residuum
        temp_2.data = -full_pre_a * temp_1
        temp_1.data += a_b_block_matrix * temp_2

        alpha = previous_residual_error_squared / \
            InnerProduct(full_preconditioned_residuum, temp_1)
        solution.data += alpha * full_preconditioned_residuum
        residuum.data += (-alpha) * temp_1
        a_preconditioned_residuum.data += (-alpha) * temp_2

        temp_1.data = full_pre_schur_complement @ full_b * a_preconditioned_residuum

        current_residual_error_squared = InnerProduct(temp_1, residuum)
        beta = current_residual_error_squared / previous_residual_error_squared

        full_preconditioned_residuum *= beta
        full_preconditioned_residuum.data += temp_1

        iteration_timer.Stop()
    else:
        print("\nWarning: CG did not converge to TOL")

    print("")
    return (solution, errors)
