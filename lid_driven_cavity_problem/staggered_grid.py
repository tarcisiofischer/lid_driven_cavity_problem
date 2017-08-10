class Mesh2d(object):
    __slots__ = (
        'nx',
        'ny',
        'phi',
        'phi_old',
    )

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.phi = [0] * (self.nx * self.ny)
        self.phi_old = [0] * (self.nx * self.ny)


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
    )

    def __init__(self, size_x, size_y, nx, ny, dt, rho, mi, bc):
        self.dt = dt
        self.dx = size_x / nx
        self.dy = size_y / ny
        self.rho = rho
        self.mi = mi
        self.bc = bc

        self.pressure_mesh = Mesh2d(nx, ny)
        self.ns_x_mesh = Mesh2d(nx - 1, ny)
        self.ns_y_mesh = Mesh2d(nx, ny - 1)

    def __len__(self):
        return len(self.pressure_mesh) + len(self.ns_x_mesh) + len(self.ns_y_mesh)
