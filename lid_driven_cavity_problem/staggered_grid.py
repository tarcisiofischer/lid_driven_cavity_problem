class Node(object):
    __slots__ = (
        'dx',
        'dy',
        'u_e',
        'u_w',
        'v_n',
        'v_s',
        'rho',
        'u_P',
        'u_P_old',
        'u_E',
        'u_W',
        'u_N',
        'u_S',
        'mi',
        'beta_u_e',
        'beta_u_w',
        'beta_v_n',
        'beta_v_s',
        'P_e',
        'P_w',
        'v_P',
        'v_P_old',
        'v_E',
        'v_W',
        'v_N',
        'v_S',
        'dv_e_dx',
        'dv_w_dx',
        'dv_n_dx',
        'dv_s_dx',
    )

class Graph(object):
    __slots__ = (
        'dt',
        'nodes',
    )
