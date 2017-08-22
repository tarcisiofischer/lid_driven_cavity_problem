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


def _calculate_jacobian_mask(nx, ny, dof):
    n_equations = n_vars = nx * ny

    j_structure = np.zeros((n_equations, n_vars), dtype=bool)
    for i in range(n_equations):
        j_structure[i, i] = 1.0

        if i - 1 >= 0:
            j_structure[i, i - 1] = 1.0
        if i + 1 < n_vars:
            j_structure[i, i + 1] = 1.0

        if i - nx >= 0:
            j_structure[i, i - nx] = 1.0
        if i + nx < n_vars:
            j_structure[i, i + nx] = 1.0

        if i - nx + 1 >= 0:
            j_structure[i, i - nx + 1] = 1.0
        if i + nx - 1 < n_vars:
            j_structure[i, i + nx - 1] = 1.0

    j_structure = np.kron(j_structure, np.ones((dof, dof)))
    return j_structure
