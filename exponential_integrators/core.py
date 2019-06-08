
from ngsolve import *
def gram_schmidt(space, tries=3):
    for tries in range(tries):
        for j in range(len(space)):
            for i in range(0, j):
                space[j].data -= InnerProduct(space[i], space[j]) / InnerProduct(space[i], space[i]) * space[i]
            # normalising
            # krylov_space[j].data = 1 / Norm(krylov_space[j]) * krylov_space[j]

    # checking the orthogonality
    # Orthogonality = Matrix(len(krylov_space), len(krylov_space))
    # for i in range(len(krylov_space)):
    #    for j in range(len(krylov_space)):
    #        Orthogonality[i, j] = InnerProduct(krylov_space[i], krylov_space[j])
    # print("orthogonality\n", numpy.round(Orthogonality,12))
    return space


def reduced_space_projection_update(space, a, m, Am, Mm):
    # timer_prep = Timer("InnerProducts")
    # timer_prep.Start()

    M_tmp = m.mat.CreateColVector()
    A_tmp = a.mat.CreateColVector()
    for j in range(len(space)):
        M_tmp.data = m.mat * space[j]
        A_tmp.data = a.mat * space[j]
        for i in range(len(space)):
            Mm[i, j] = InnerProduct(M_tmp, space[i])
            Am[i, j] = InnerProduct(A_tmp, space[i])
    # timer_prep.Stop()
    # print("time of inner products", timer_prep.time)
    return Am, Mm

def reduced_space_projection(space, matrix, matrix_small):
    # timer_prep = Timer("InnerProducts")
    # timer_prep.Start()

    M_tmp = matrix.CreateColVector()

    for j in range(len(space)):
        M_tmp.data = matrix * space[j]

        for i in range(len(space)):
            matrix_small[i, j] = InnerProduct(M_tmp, space[i])

    # timer_prep.Stop()
    # print("time of inner products", timer_prep.time)
    return matrix_small