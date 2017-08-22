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
    from scipy.sparse import diags, kron

    N = nx * ny

    j_structure = diags(
        np.ones(shape=(7,)),
        [-nx + 1, -nx, -1, 0, 1, +nx, +nx - 1],
        shape=(N, N),
        format='coo',
    )

    j_structure = kron(
        j_structure,
        np.ones(shape=(dof, dof)),
        format='coo',
    )

    return j_structure
