import numpy as np


def residual_function(X, graph):
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh
    residual = np.full(shape=(len(X),), fill_value=np.inf)

    # Knowns (Constants)
    dt = graph.dt
    dx = graph.dx
    dy = graph.dy
    rho = graph.rho
    mi = graph.mi
    bc = graph.bc

    # Extract unknown vectors
    P = X[0::3]

    extended_U = X[1::3]
    nx = graph.ns_x_mesh.nx
    ny = graph.ns_x_mesh.ny
    U_idxs = np.arange(0, len(graph.ns_x_mesh)) + np.repeat(np.arange(0, ny), nx)
    U = extended_U[U_idxs]

    V = X[2::3][0:len(ns_y_mesh)]

    # Equation for Conservation of Mass
    # =============================================================================================
    i = np.arange(len(pressure_mesh))
    j = i // pressure_mesh.nx

    # Index conversion
    i_U_w = i - j - 1
    i_U_e = i_U_w + 1
    i_V_n = np.array(i)
    i_V_s = i_V_n - pressure_mesh.nx
    # Remove invalid indexes by setting them to 0. Note that those indexes shouldn't be used at all.
    i_U_e[np.logical_or(i_U_w >= len(ns_x_mesh), i_U_w < 0)] = 0
    i_U_e[np.logical_or(i_U_e >= len(ns_x_mesh), i_U_e < 0)] = 0
    i_V_n[np.logical_or(i_V_n >= len(ns_y_mesh), i_V_n < 0)] = 0
    i_V_s[np.logical_or(i_V_s >= len(ns_y_mesh), i_V_s < 0)] = 0

    # Knowns
    is_left_boundary = i % pressure_mesh.nx == 0
    is_right_boundary = (i + 1) % pressure_mesh.nx == 0
    is_bottom_boundary = j % pressure_mesh.ny == 0
    is_top_boundary = (j + 1) % pressure_mesh.ny == 0

    # Unknowns
    U_w = np.zeros(shape=(len(pressure_mesh)))
    U_e = np.zeros(shape=(len(pressure_mesh)))
    V_n = np.zeros(shape=(len(pressure_mesh)))
    V_s = np.zeros(shape=(len(pressure_mesh)))

    U_w[~is_left_boundary] = U[i_U_w][~is_left_boundary]
    U_e[~is_right_boundary] = U[i_U_e][~is_right_boundary]
    V_n[~is_top_boundary] = V[i_V_n][~is_top_boundary]
    V_s[~is_bottom_boundary] = V[i_V_s][~is_bottom_boundary]

    # Residual equation for Conservation of Mass
    ii = 3 * i
    residual[ii] = (U_e * dy - U_w * dy) + (V_n * dx - V_s * dx)

    # Navier Stokes (X)
    #===============================================================================================
    i = np.arange(len(ns_x_mesh))
    j = i // ns_x_mesh.nx

    # Index conversion
    i_U_P = np.array(i)
    i_U_W = i - 1
    i_U_E = i + 1
    i_U_N = i + ns_x_mesh.nx
    i_U_S = i - ns_x_mesh.nx
    i_P_w = i + (i // ns_x_mesh.nx)
    i_P_e = i_P_w + 1
    i_V_NW = i + j
    i_V_NE = i_V_NW + 1
    i_V_SW = i_V_NW - ns_y_mesh.nx
    i_V_SE = i_V_NE - ns_y_mesh.nx
    # Remove invalid indexes by setting them to 0. Note that those indexes shouldn't be used at all.
    i_U_P [np.logical_or(i_U_P >= len(ns_x_mesh), i_U_P < 0)] = 0
    i_U_W [np.logical_or(i_U_W >= len(ns_x_mesh), i_U_W < 0)] = 0
    i_U_E [np.logical_or(i_U_E >= len(ns_x_mesh), i_U_E < 0)] = 0
    i_U_N [np.logical_or(i_U_N >= len(ns_x_mesh), i_U_N < 0)] = 0
    i_U_S [np.logical_or(i_U_S >= len(ns_x_mesh), i_U_S < 0)] = 0
    i_P_w [np.logical_or(i_P_w >= len(pressure_mesh), i_P_w < 0)] = 0
    i_P_e [np.logical_or(i_P_e >= len(pressure_mesh), i_P_e < 0)] = 0
    i_V_NW[np.logical_or(i_V_NW >= len(ns_y_mesh), i_V_NW < 0)] = 0
    i_V_NE[np.logical_or(i_V_NE >= len(ns_y_mesh), i_V_NE < 0)] = 0
    i_V_SW[np.logical_or(i_V_SW >= len(ns_y_mesh), i_V_SW < 0)] = 0
    i_V_SE[np.logical_or(i_V_SE >= len(ns_y_mesh), i_V_SE < 0)] = 0

    # Knowns
    U_P_old = np.array(ns_x_mesh.phi_old)[i_U_P]
    is_left_boundary = i % ns_x_mesh.nx == 0
    is_right_boundary = (i + 1) % ns_x_mesh.nx == 0
    is_bottom_boundary = j % ns_x_mesh.ny == 0
    is_top_boundary = (j + 1) % ns_x_mesh.ny == 0

    # Unknowns
    U_P = np.zeros(shape=(len(ns_x_mesh)),)
    U_W = np.zeros(shape=(len(ns_x_mesh)),)
    U_E = np.zeros(shape=(len(ns_x_mesh)),)
    U_N = np.zeros(shape=(len(ns_x_mesh)),)
    U_S = np.zeros(shape=(len(ns_x_mesh)),)
    P_w = np.zeros(shape=(len(ns_x_mesh)),)
    P_e = np.zeros(shape=(len(ns_x_mesh)),)
    V_NE = np.zeros(shape=(len(ns_x_mesh)),)
    V_NW = np.zeros(shape=(len(ns_x_mesh)),)
    V_SE = np.zeros(shape=(len(ns_x_mesh)),)
    V_SW = np.zeros(shape=(len(ns_x_mesh)),)

    U_P[:] = U[i_U_P]
    U_W[~is_left_boundary] = U[i_U_W][~is_left_boundary]
    U_E[~is_right_boundary] = U[i_U_E][~is_right_boundary]
    U_N[is_top_boundary] = bc
    U_N[~is_top_boundary] = U[i_U_N][~is_top_boundary]
    U_S[~is_bottom_boundary] = U[i_U_S][~is_bottom_boundary]
    P_w[:] = P[i_P_w]
    P_e[:] = P[i_P_e]
    V_NE[~is_top_boundary] = V[i_V_NE][~is_top_boundary]
    V_NW[~is_top_boundary] = V[i_V_NW][~is_top_boundary]
    V_SE[~is_bottom_boundary] = V[i_V_SE][~is_bottom_boundary]
    V_SW[~is_bottom_boundary] = V[i_V_SW][~is_bottom_boundary]

    # Calculated (Interpolated and Secondary Variables)
    dU_e_dx = (U_E - U_P) / dx
    dU_w_dx = (U_P - U_W) / dx
    dU_n_dx = (U_N - U_P) / dy
    dU_s_dx = (U_P - U_S) / dy
    U_e = (U_E + U_P) / 2.0
    U_w = (U_P + U_W) / 2.0
    V_n = (V_NE + V_NW) / 2.0
    V_s = (V_SE + V_SW) / 2.0
    beta_U_e = np.where(U_e > 0.0, 0.5, -0.5)
    beta_U_w = np.where(U_w > 0.0, 0.5, -0.5)
    beta_V_n = np.where(V_n > 0.0, 0.5, -0.5)
    beta_V_s = np.where(V_s > 0.0, 0.5, -0.5)

    # Terms calculation for Navier Stokes X
    transient_term = (rho * U_P - rho * U_P_old) * (dx * dy / dt)
    advective_term = \
        rho * U_e * ((.5 - beta_U_e) * U_E + (.5 + beta_U_e) * U_P) * dy - \
        rho * U_w * ((.5 - beta_U_w) * U_P + (.5 + beta_U_w) * U_W) * dy + \
        rho * V_n * ((.5 - beta_V_n) * U_N + (.5 + beta_V_n) * U_P) * dx - \
        rho * V_s * ((.5 - beta_V_s) * U_P + (.5 + beta_V_s) * U_S) * dx
    difusive_term = \
        mi * dU_e_dx * dy - \
        mi * dU_w_dx * dy + \
        mi * dU_n_dx * dx - \
        mi * dU_s_dx * dx
    source_term = -(P_e - P_w) * dy

    # Residual equation for Navier Stokes X
    ii = 3 * (i + j) + 1
    residual[ii] = transient_term + advective_term - difusive_term - source_term

    # Navier Stokes (Y)
    #===============================================================================================
    i = np.arange(len(ns_y_mesh))
    j = i // ns_y_mesh.nx

    # Index conversion
    i_V_P = np.array(i)
    i_V_W = i - 1
    i_V_E = i + 1
    i_V_N = i + ns_y_mesh.nx
    i_V_S = i - ns_y_mesh.nx
    i_P_n = i + ns_y_mesh.nx
    i_P_s = np.array(i)
    i_U_SE = i - j
    i_U_SW = i_U_SE - 1
    i_U_NE = i_U_SE + ns_x_mesh.nx
    i_U_NW = i_U_SW + ns_x_mesh.nx
    # Remove invalid indexes by setting them to 0. Note that those indexes shouldn't be used at all.
    i_V_P [np.logical_or(i_V_P >= len(ns_y_mesh), i_V_P < 0)] = 0
    i_V_W [np.logical_or(i_V_W >= len(ns_y_mesh), i_V_W < 0)] = 0
    i_V_E [np.logical_or(i_V_E >= len(ns_y_mesh), i_V_E < 0)] = 0
    i_V_N [np.logical_or(i_V_N >= len(ns_y_mesh), i_V_N < 0)] = 0
    i_V_S [np.logical_or(i_V_S >= len(ns_y_mesh), i_V_S < 0)] = 0
    i_P_n [np.logical_or(i_P_n >= len(pressure_mesh), i_P_n < 0)] = 0
    i_P_s [np.logical_or(i_P_s >= len(pressure_mesh), i_P_s < 0)] = 0
    i_U_SE[np.logical_or(i_U_SE >= len(ns_x_mesh), i_U_SE < 0)] = 0
    i_U_SW[np.logical_or(i_U_SW >= len(ns_x_mesh), i_U_SW < 0)] = 0
    i_U_NE[np.logical_or(i_U_NE >= len(ns_x_mesh), i_U_NE < 0)] = 0
    i_U_NW[np.logical_or(i_U_NW >= len(ns_x_mesh), i_U_NW < 0)] = 0

    # Knowns
    V_P_old = np.array(ns_y_mesh.phi_old)[i_V_P]
    is_left_boundary = i % ns_y_mesh.nx == 0
    is_right_boundary = (i + 1) % ns_y_mesh.nx == 0
    is_bottom_boundary = j % ns_y_mesh.ny == 0
    is_top_boundary = (j + 1) % ns_y_mesh.ny == 0

    # Unknowns
    V_P = np.zeros(shape=(len(ns_y_mesh),))
    V_E = np.zeros(shape=(len(ns_y_mesh),))
    V_W = np.zeros(shape=(len(ns_y_mesh),))
    V_N = np.zeros(shape=(len(ns_y_mesh),))
    V_S = np.zeros(shape=(len(ns_y_mesh),))
    P_n = np.zeros(shape=(len(ns_y_mesh),))
    P_s = np.zeros(shape=(len(ns_y_mesh),))
    U_SE = np.zeros(shape=(len(ns_y_mesh),))
    U_SW = np.zeros(shape=(len(ns_y_mesh),))
    U_NE = np.zeros(shape=(len(ns_y_mesh),))
    U_NW = np.zeros(shape=(len(ns_y_mesh),))

    V_P[:] = V[i_V_P]
    V_E[~is_right_boundary] = V[i_V_E][~is_right_boundary]
    V_W[~is_left_boundary] = V[i_V_W][~is_left_boundary]
    V_N[~is_top_boundary] = V[i_V_N][~is_top_boundary]
    V_S[~is_bottom_boundary] = V[i_V_S][~is_bottom_boundary]
    P_n[:] = P[i_P_n]
    P_s[:] = P[i_P_s]
    U_SE[~is_right_boundary] = U[i_U_SE][~is_right_boundary]
    U_SW[~is_left_boundary] = U[i_U_SW][~is_left_boundary]
    U_NE[~is_right_boundary] = U[i_U_NE][~is_right_boundary]
    U_NE[is_top_boundary] = bc
    U_NE[is_right_boundary] = 0.0
    U_NW[~is_left_boundary] = U[i_U_NW][~is_left_boundary]
    U_NW[is_top_boundary] = bc
    U_NW[is_left_boundary] = 0.0

    # Calculated (Interpolated and Secondary Variables)
    dV_e_dx = (V_E - V_P) / dx
    dV_w_dx = (V_P - V_W) / dx
    dV_n_dx = (V_N - V_P) / dy
    dV_s_dx = (V_P - V_S) / dy
    U_e = (U_NE + U_SE) / 2.0
    U_w = (U_NW + U_SW) / 2.0
    V_n = (V_P + V_N) / 2.0
    V_s = (V_S + V_P) / 2.0
    beta_U_e = np.where(U_e > 0.0, 0.5, -0.5)
    beta_U_w = np.where(U_w > 0.0, 0.5, -0.5)
    beta_V_n = np.where(V_n > 0.0, 0.5, -0.5)
    beta_V_s = np.where(V_s > 0.0, 0.5, -0.5)

    # Term calculations for Navier Stokes Y
    transient_term = (rho * V_P - rho * V_P_old) * (dx * dy / dt)
    advective_term = \
        rho * U_e * ((.5 - beta_U_e) * V_E + (.5 + beta_U_e) * V_P) * dy - \
        rho * U_w * ((.5 - beta_U_w) * V_P + (.5 + beta_U_w) * V_W) * dy + \
        rho * V_n * ((.5 - beta_V_n) * V_N + (.5 + beta_V_n) * V_P) * dx - \
        rho * V_s * ((.5 - beta_V_s) * V_P + (.5 + beta_V_s) * V_S) * dx
    difusive_term = \
        mi * dV_e_dx * dy - \
        mi * dV_w_dx * dy + \
        mi * dV_n_dx * dx - \
        mi * dV_s_dx * dx
    source_term = -(P_n - P_s) * dx

    # Residual equation for Navier Stokes Y
    ii = 3 * i + 2
    residual[ii] = transient_term + advective_term - difusive_term - source_term

    # Remaining functions (If any)
    #===============================================================================================
    # Set all remaining residuals with x[i] - x[i] = R
    unused_functions_idxs = np.where(residual == np.inf)
    residual[unused_functions_idxs] = X[unused_functions_idxs]

    return residual
