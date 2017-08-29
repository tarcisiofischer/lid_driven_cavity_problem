class Mesh2d(object):
    __slots__ = (
        'nx',
        'ny',
        'phi',
        'phi_old',
    )

    def __init__(self, nx, ny, initial_phi):
        self.nx = nx
        self.ny = ny

        import numpy as np
        self.phi = np.array([initial_phi] * (self.nx * self.ny))
        self.phi_old = np.array([initial_phi] * (self.nx * self.ny))


    def __len__(self):
        return self.nx * self.ny


class Graph(object):
    __slots__ = (
        'dt',
        'dx',
        'dy',
        'rho',
        'mi',
        'pressure_mesh',
        'ns_x_mesh',
        'ns_y_mesh',
        'bc',
        'use_cds',
    )

    def __init__(self, size_x, size_y, nx, ny, dt, rho, mi, bc, initial_P=1.0, initial_U=1.0, initial_V=1.0):
        self.dt = dt
        self.dx = size_x / nx
        self.dy = size_y / ny
        self.rho = rho
        self.mi = mi
        self.bc = bc
        self.use_cds = False

        self.pressure_mesh = Mesh2d(nx, ny, initial_P)
        self.ns_x_mesh = Mesh2d(nx - 1, ny, initial_U)
        self.ns_y_mesh = Mesh2d(nx, ny - 1, initial_V)

    def __len__(self):
        return len(self.pressure_mesh) + len(self.ns_x_mesh) + len(self.ns_y_mesh)
