def residual_function(U, V, P, graph):
    # Notes:
    #
    # Think about using petsc4py's DMDA so that we don't need to worry about
    # the Jacobian matrix.
    #
    unknowns = 3
    residual = [None] * len(graph) * unknowns
    mass_equation_offset = len(graph) * 0
    ns_x_equation_offset = len(graph) * 1
    ns_y_equation_offset = len(graph) * 2

    dt = graph.dt

    for i, node in range(len(graph)):
        # Constants
        dx = node.dx
        dy = node.dy
        rho = node.rho
        mi = node.mi
        U_P_old = node.u_P_old
        V_P_old = node.v_P_old

        # Unknowns
        U_P = node.U_P_from(U)
        U_E = node.U_E_from(U)
        U_W = node.U_W_from(U)
        U_N = node.U_N_from(U)
        U_S = node.U_S_from(U)
        V_P = node.V_P_from(V)
        V_E = node.V_E_from(V)
        V_SE = node.V_SE_from(V)
        V_W = node.V_W_from(V)
        V_N = node.V_N_from(V)
        V_S = node.V_S_from(V)
        P_P = node.P_P_from(P)
        P_E = node.P_E_from(P)
        P_W = node.P_W_from(P)

        # Calculated
        u_e = (U_E + U_P) / 2.0
        u_w = (U_P + U_W) / 2.0
        v_n = (V_E + V_P) / 2.0
        v_s = (V_S + V_SE) / 2.0
        beta_u_e = node.beta_u_e
        beta_u_w = node.beta_u_w
        beta_v_n = node.beta_v_n
        beta_v_s = node.beta_v_s
        P_e = node.P_e
        P_w = node.P_w
        du_e_dx = (U_E - U_P) / dx
        du_w_dx = (U_P - U_W) / dx
        du_n_dx = (U_N - U_P) / dy
        du_s_dx = (U_P - U_S) / dy
        dv_e_dx = (V_E - V_P) / dx
        dv_w_dx = (V_P - V_W) / dx
        dv_n_dx = (V_N - V_P) / dy
        dv_s_dx = (V_P - V_S) / dy

        # Conservation of Mass
        ii = mass_equation_offset * len(graph) + i
        residual[ii] = (u_e * dy - u_w * dy) + (v_n * dx - v_s * dx)
 
        # Navier Stokes X 
        transient_term = (rho * U_P - rho * U_P_old) * (dx * dy / dt)
        advective_term = \
            rho * u_e * ((.5 - beta_u_e) * U_E + (.5 + beta_u_e) * U_P) * dy - \
            rho * u_w * ((.5 - beta_u_w) * U_P + (.5 + beta_u_w) * U_W) * dy + \
            rho * v_n * ((.5 - beta_v_n) * U_N + (.5 + beta_v_n) * U_P) * dx - \
            rho * v_s * ((.5 - beta_v_s) * U_P + (.5 + beta_v_s) * U_S) * dx
        difusive_term  = \
            mi * du_e_dx * dy - \
            mi * du_w_dx * dy + \
            mi * du_n_dx * dx - \
            mi * du_s_dx * dx
        source_term    = -(P_e - P_w) * dy 

        ii = ns_x_equation_offset * len(graph) + i
        residual[i] = transient_term + advective_term - difusive_term - source_term

        # Navier Stokes Y
        transient_term = (rho * V_P - rho * V_P_old) * (dx * dy / dt)
        advective_term = \
            rho * u_e * ((.5 - beta_u_e) * V_E + (.5 + beta_u_e) * V_P) * dy - \
            rho * u_w * ((.5 - beta_u_w) * V_P + (.5 + beta_u_w) * V_W) * dy + \
            rho * v_n * ((.5 - beta_v_n) * V_N + (.5 + beta_v_n) * V_P) * dx - \
            rho * v_s * ((.5 - beta_v_s) * V_P + (.5 + beta_v_s) * V_S) * dx
        difusive_term  = \
            mi * dv_e_dx * dy - \
            mi * dv_w_dx * dy + \
            mi * dv_n_dx * dx - \
            mi * dv_s_dx * dx
        source_term    = -(node.P_n - node.P_s) * dy 

        ii = ns_y_equation_offset * len(graph) + i
        residual[ii] = transient_term + advective_term - difusive_term - source_term

    # Sanity check
    assert None not in residual, 'Missing equation in residual function'

    return residual
