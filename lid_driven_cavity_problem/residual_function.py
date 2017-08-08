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
    dx = graph.dx
    dy = graph.dy
    rho = graph.rho
    mi = graph.mi
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh

    for i, pressure_node in range(len(pressure_mesh)):
        # Knowns
        U_e = pressure_node.U_e
        U_w = pressure_node.U_w
        V_n = pressure_node.V_n
        V_s = pressure_node.V_s

        # Conservation of Mass
        ii = mass_equation_offset * len(graph) + i
        residual[ii] = (U_e * dy - U_w * dy) + (V_n * dx - V_s * dx)

    for i, ns_x_node in range(len(ns_x_mesh)):
        # Knowns
        U_P_old = ns_x_node.U_P_old
        U_P = ns_x_node.U_P
        U_E = ns_x_node.U_E
        U_W = ns_x_node.U_W
        U_N = ns_x_node.U_N
        U_S = ns_x_node.U_S
        P_e = ns_x_node.P_e
        P_w = ns_x_node.P_w

        # Calculated
        dU_e_dx = (U_E - U_P) / dx
        dU_w_dx = (U_P - U_W) / dx
        dU_n_dx = (U_N - U_P) / dy
        dU_s_dx = (U_P - U_S) / dy
        beta_U_e = ns_x_node.beta_U_e
        beta_U_w = ns_x_node.beta_U_w
        beta_V_n = ns_x_node.beta_V_n
        beta_V_s = ns_x_node.beta_V_s
        U_e = ns_x_node.U_e
        U_w = ns_x_node.U_w
        V_n = ns_x_node.V_n
        V_s = ns_x_node.V_s

        # Navier Stokes X
        transient_term = (rho * U_P - rho * U_P_old) * (dx * dy / dt)
        advective_term = \
            rho * U_e * ((.5 - beta_U_e) * U_E + (.5 + beta_U_e) * U_P) * dy - \
            rho * U_w * ((.5 - beta_U_w) * U_P + (.5 + beta_U_w) * U_W) * dy + \
            rho * V_n * ((.5 - beta_V_n) * U_N + (.5 + beta_V_n) * U_P) * dx - \
            rho * V_s * ((.5 - beta_V_s) * U_P + (.5 + beta_V_s) * U_S) * dx
        difusive_term  = \
            mi * dU_e_dx * dy - \
            mi * dU_w_dx * dy + \
            mi * dU_n_dx * dx - \
            mi * dU_s_dx * dx
        source_term    = -(P_e - P_w) * dy 

        ii = ns_x_equation_offset * len(graph) + i
        residual[i] = transient_term + advective_term - difusive_term - source_term

    for i, ns_y_node in range(len(ns_y_mesh)):
        # Knowns
        V_P_old = ns_y_node.V_P_old
        V_P = ns_y_node.V_P
        V_E = ns_y_node.V_E
        V_W = ns_y_node.V_W
        V_N = ns_y_node.V_N
        V_S = ns_y_node.V_S
        P_n = ns_y_node.P_n
        P_s = ns_y_node.P_s

        # Calculated
        beta_U_e = ns_y_node.beta_U_e
        beta_U_w = ns_y_node.beta_U_w
        beta_V_n = ns_y_node.beta_V_n
        beta_V_s = ns_y_node.beta_V_s
        dV_e_dx = (V_E - V_P) / dx
        dV_w_dx = (V_P - V_W) / dx
        dV_n_dx = (V_N - V_P) / dy
        dV_s_dx = (V_P - V_S) / dy
        U_e = ns_y_node.U_e
        U_w = ns_y_node.U_w
        V_n = ns_y_node.V_n
        V_s = ns_y_node.V_s

        # Navier Stokes Y
        transient_term = (rho * V_P - rho * V_P_old) * (dx * dy / dt)
        advective_term = \
            rho * U_e * ((.5 - beta_U_e) * V_E + (.5 + beta_U_e) * V_P) * dy - \
            rho * U_w * ((.5 - beta_U_w) * V_P + (.5 + beta_U_w) * V_W) * dy + \
            rho * V_n * ((.5 - beta_V_n) * V_N + (.5 + beta_V_n) * V_P) * dx - \
            rho * V_s * ((.5 - beta_V_s) * V_P + (.5 + beta_V_s) * V_S) * dx
        difusive_term  = \
            mi * dV_e_dx * dy - \
            mi * dV_w_dx * dy + \
            mi * dV_n_dx * dx - \
            mi * dV_s_dx * dx
        source_term    = -(P_n - P_s) * dy 

        ii = ns_y_equation_offset * len(graph) + i
        residual[ii] = transient_term + advective_term - difusive_term - source_term

    # Sanity check
    assert None not in residual, 'Missing equation in residual function'

    return residual
