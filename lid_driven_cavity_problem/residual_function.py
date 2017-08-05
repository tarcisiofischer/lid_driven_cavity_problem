def residual_function(graph):
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

    # Conservation of Mass
    # u_e * dy - u_w * dy + v_n * dx - v_s * dx
    start = mass_equation_offset * len(graph)
    for i, node in range(start, start + len(graph)):
        dx = node.dx
        dy = node.dy
        u_e = node.u_e
        u_w = node.u_w
        v_n = node.v_n
        v_s = node.v_s
  
        residual[i] = (u_e * dy - u_w * dy) + (v_n * dx - v_s * dx)
 
    # Navier Stokes X 
    start = ns_x_equation_offset * len(graph)
    for i, node in range(start, start + len(graph)):
        dx = node.dx
        dy = node.dy
        dt = graph.dt
        rho = node.rho
        u_P = node.u_P
        u_P_old = node.u_P_old
        u_E = node.u_E
        u_W = node.u_W
        u_N = node.u_N
        u_S = node.u_S
        u_e = node.u_e
        u_w = node.u_w
        v_n = node.v_n
        v_s = node.v_s
        mi = node.mi
        beta_u_e = node.beta_u_e
        beta_u_w = node.beta_u_w
        beta_v_n = node.beta_v_n
        beta_v_s = node.beta_v_s
        P_e = node.P_e
        P_w = node.P_w

        transient_term = (rho * u_P - rho * u_P_old) * (dx * dy / dt)
        advective_term = \
            rho * u_e * ((.5 - beta_u_e) * u_E + (.5 + beta_u_e) * u_P) * dy - \
            rho * u_w * ((.5 - beta_u_w) * u_P + (.5 + beta_u_w) * u_W) * dy + \
            rho * v_n * ((.5 - beta_v_n) * u_N + (.5 + beta_v_n) * u_P) * dx - \
            rho * v_s * ((.5 - beta_v_s) * u_P + (.5 + beta_v_s) * u_S) * dx
        difusive_term  = \
            mi * (u_E - u_P) / dx * dy - \
            mi * (u_P - u_W) / dx * dy + \
            mi * (u_N - u_P) / dy * dx - \
            mi * (u_P - u_S) / dy * dx
        source_term    = -(P_e - P_w) * dy 

        residual[i] = transient_term + advective_term - difusive_term - source_term

    # Navier Stokes Y
    start = ns_y_equation_offset * len(graph)
    for i, node in range(start, start + len(graph)):
        dx = node.dx
        dy = node.dy
        dt = graph.dt
        rho = node.rho
        v_P = node.u_P
        v_P_old = node.u_P_old
        v_E = node.u_E
        v_W = node.u_W
        v_N = node.u_N
        v_S = node.u_S
        u_e = node.u_e
        u_w = node.u_w
        v_n = node.v_n
        v_s = node.v_s
        mi = node.mi
        beta_u_e = node.beta_u_e
        beta_u_w = node.beta_u_w
        beta_v_n = node.beta_v_n
        beta_v_s = node.beta_v_s
        dv_e_dx = node.dv_e_dx # (v_E - v_P) / dx
        dv_w_dx = node.dv_w_dx # (v_P - v_W) / dx
        dv_n_dx = node.dv_n_dx # (v_N - v_P) / dy
        dv_s_dx = node.dv_s_dx # (v_P - v_S) / dy

        transient_term = (rho * v_P - rho * v_P_old) * (dx * dy / dt)
        advective_term = \
            rho * u_e * ((.5 - beta_u_e) * v_E + (.5 + beta_u_e) * v_P) * dy - \
            rho * u_w * ((.5 - beta_u_w) * v_P + (.5 + beta_u_w) * v_W) * dy + \
            rho * v_n * ((.5 - beta_v_n) * v_N + (.5 + beta_v_n) * v_P) * dx - \
            rho * v_s * ((.5 - beta_v_s) * v_P + (.5 + beta_v_s) * v_S) * dx
        difusive_term  = \
            mi * dv_e_dx * dy - \
            mi * dv_w_dx * dy + \
            mi * dv_n_dx * dx - \
            mi * dv_s_dx * dx
        source_term    = -(node.P_n - node.P_s) * dy 

        residual[i] = transient_term + advective_term - difusive_term - source_term

    # Sanity check
    assert None not in residual, 'Missing equation in residual function'

    return residual
