from lid_driven_cavity_problem.staggered_grid import Graph
from lid_driven_cavity_problem.newton_solver import solve
from lid_driven_cavity_problem.time_stepper import run_simulation
import matplotlib.pyplot as plt
import numpy as np


def test_run_solver():
    size_x = 0.1
    size_y = 0.1
    nx = 15
    ny = 15
    dt = 1e-1
    rho = 0.1
    mi = 0.1
    final_time = 0.5
    bc = 200
    graph = Graph(size_x, size_y, nx, ny, dt, rho, mi, bc)
    result = run_simulation(graph, final_time)
    U = np.array(result.ns_x_mesh.phi)
    V = np.array(result.ns_y_mesh.phi)

#     np.savetxt('U.txt', np.array(U))
#     np.savetxt('V.txt', np.array(V))
#     U = np.loadtxt('U.txt')
#     V = np.loadtxt('U.txt')

    U = U.reshape(nx, ny - 1)
    V = V.reshape(nx - 1, ny)

    U = np.c_[[0.0] * nx, U, [0.0] * ny]
    U = (U[:,1:] + U[:,:-1]) / 2.0

    V = np.r_[[[0.0] * nx], V, [[0.0] * ny]]
    V = (V[1:,:] + V[:-1,:]) / 2.0

    X, Y = np.meshgrid(np.arange(0.0, size_x, size_x / nx), np.arange(0.0, size_y, size_y / ny))
    plt.figure()
    plt.quiver(X, Y, U, V)
    plt.show()
