from ngsolve import *
from ngsolve.ngstd import Timer


def orthonormalize(basis, tries=3):
    orthonormalize_timer = Timer("orthonormalization")
    orthonormalize_timer.Start()
    for tries in range(tries):
        for j in range(len(basis)):
            for i in range(j):
                basis[j].data -= InnerProduct(basis[i], basis[j]) / \
                    InnerProduct(basis[i], basis[i]) * basis[i]
            basis[j].data = 1 / Norm(basis[j]) * basis[j]
    orthonormalize_timer.Stop()

    return basis
