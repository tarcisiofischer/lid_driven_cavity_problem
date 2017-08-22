from scipy.optimize.slsqp import approx_jacobian

from lid_driven_cavity_problem import residual_function
import numpy as np


def _create_X(U, V, P, graph):
    X = np.zeros(shape=(3 * len(P),))
    X[0::3] = P

    extended_U = np.ones(shape=(len(P),))
    nx = graph.ns_x_mesh.nx
    ny = graph.ns_x_mesh.ny
    U_idxs = np.arange(0, len(graph.ns_x_mesh)) + np.repeat(np.arange(0, ny), nx)
    extended_U[U_idxs] = U
    X[1::3] = extended_U

    extended_V = np.r_[V, np.ones(shape=(len(P) - len(V),))]
    X[2::3] = extended_V

    return X


def _recover_X(X, graph):
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh

    P = X[0::3][0:len(pressure_mesh)]

    extended_U = X[1::3]
    nx = ns_x_mesh.nx
    ny = ns_x_mesh.ny
    U_idxs = np.arange(0, len(ns_x_mesh)) + np.repeat(np.arange(0, ny), nx)
    U = extended_U[U_idxs]

    V = X[2::3][0:len(ns_y_mesh)]

    return U, V, P


__j_structure_cache = None
def _calculate_jacobian_mask(N, graph):
    COMPARE_JACOBIAN_AND_SHOW = False

    global __j_structure_cache
    if __j_structure_cache is not None:
        return __j_structure_cache

    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh

    # Must have values in diagonal because of ILU preconditioner
    j_structure__new = np.zeros((N // 3, N // 3), dtype=bool)

    for i in range(N // 3):
        j_structure__new[i, i] = 1.0
        if i - 1 >= 0:
            j_structure__new[i, i - 1] = 1.0
        if i + 1 < N // 3:
            j_structure__new[i, i + 1] = 1.0
        if i - pressure_mesh.nx >= 0:
            j_structure__new[i, i - pressure_mesh.nx] = 1.0
        if i + pressure_mesh.nx < N // 3:
            j_structure__new[i, i + pressure_mesh.nx] = 1.0
        if i - pressure_mesh.nx + 1 >= 0:
            j_structure__new[i, i - pressure_mesh.nx + 1] = 1.0
        if i + pressure_mesh.nx - 1 < N // 3:
            j_structure__new[i, i + pressure_mesh.nx - 1] = 1.0

    j_structure__new = np.kron(j_structure__new, np.ones((3, 3)))

    if COMPARE_JACOBIAN_AND_SHOW:
        j_structure = np.zeros((N, N), dtype=bool)
        # Creates the Jacobian 4 times, because of upwind asymmetries
        for u_sign in [-1.0, 1.0]:
            for v_sign in [-1.0, 1.0]:
                fake_U = [u_sign * 1e-2 * (i + 1) ** 2 for i in range(len(ns_x_mesh.phi))]
                fake_V = [v_sign * 1e-2 * (i + 1) ** 2 for i in range(len(ns_y_mesh.phi))]
                fake_P = [1e3 * (i + 1) ** 2 for i in range(len(pressure_mesh.phi))]
                fake_X = _create_X(fake_U, fake_V, fake_P, graph)
                current_j_structure = approx_jacobian(fake_X, residual_function, 1e-4, graph).astype(dtype='bool')
                j_structure = np.bitwise_or(j_structure, current_j_structure)
        cmp = j_structure + j_structure__new
        import matplotlib.pyplot as plt
        plt.imshow(cmp, aspect='equal', interpolation='none')
        plt.show()

    j_structure = j_structure__new

    __j_structure_cache = j_structure
    return j_structure
