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

    # Staggered grids
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh

    # Knowns (Constants)
    dt = graph.dt
    dx = graph.dx
    dy = graph.dy
    rho = graph.rho
    mi = graph.mi

    for i, pressure_node in range(len(pressure_mesh)):
        # Index conversion
        i_U_w = i - (i // pressure_mesh.nx) - 1
        i_U_e = i_U_w + 1
        i_V_n = i
        i_V_s = i - (i // pressure_mesh.nx) * pressure_mesh.nx

        # Unknowns
        U_w = U[i_U_w]
        U_e = U[i_U_e]
        V_n = V[i_V_n]
        V_s = V[i_V_s]

        # Conservation of Mass
        ii = mass_equation_offset * len(graph) + i
        residual[ii] = (U_e * dy - U_w * dy) + (V_n * dx - V_s * dx)

    for i, ns_x_node in range(len(ns_x_mesh)):
        # Index conversion
        i_U_P = i
        i_U_W = i - 1
        i_U_E = i + 1
        i_U_N = i + ns_x_mesh.nx
        i_U_S = i - ns_x_mesh.nx
        i_P_w = i + (i // ns_x_mesh.nx)
        i_P_e = i_P_w + 1

        # Knowns
        U_P_old = ns_x_node.U_P_old

        # Unknowns
        U_P = U[i_U_P]
        U_W = U[i_U_W]
        U_E = U[i_U_E]
        U_N = U[i_U_N]
        U_S = U[i_U_S]
        P_w = P[i_P_w]
        P_e = P[i_P_e]

        # Calculated (Interpolated and Secondary Variables)
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
        # Index conversion
        i_V_P = i
        i_V_W = i - 1
        i_V_E = i + 1
        i_V_N = i + ns_y_mesh.nx
        i_V_S = i - ns_y_mesh.nx
        i_P_n = i + ns_y_mesh.nx
        i_P_s = i

        # Knowns
        V_P_old = ns_y_node.V_P_old

        # Unknowns
        V_P = V[i_V_P]
        V_E = V[i_V_E]
        V_W = V[i_V_W]
        V_N = V[i_V_N]
        V_S = V[i_V_S]
        P_n = P[i_P_n]
        P_s = P[i_P_s]

        # Calculated (Interpolated and Secondary Variables)
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
