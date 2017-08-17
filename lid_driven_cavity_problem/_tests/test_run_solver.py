from lid_driven_cavity_problem.staggered_grid import Graph
from lid_driven_cavity_problem.time_stepper import run_simulation
import matplotlib.pyplot as plt
import numpy as np

LOAD_RESULTS = False
SAVE_RESULTS = False
PLOT_RESULTS = True

def test_run_solver():
    print("-")
    print("TEST_RUN_SOLVER")
    
    if LOAD_RESULTS:
        U = np.loadtxt('U.txt')
        V = np.loadtxt('V.txt')
    else:
        size_x = 1.0
        size_y = 1.0
        nx = 12
        ny = 12
        dt = 1e-2
        rho = 1.0
        final_time = 0.1
        mi = 1.0
        Re = 10
        U_bc = (mi * Re) / (rho * size_x)
        print("Run Parameters:")
        print("size_x = %s" % (size_x,))
        print("size_y = %s" % (size_y,))
        print("nx = %s" % (nx,))
        print("ny = %s" % (ny,))
        print("dt = %s" % (dt,))
        print("rho = %s" % (rho,))
        print("mi = %s" % (mi,))
        print("U_bc = %s" % (U_bc,))
        print("Re = %s" % (Re,))
        print("")
     
        graph = Graph(size_x, size_y, nx, ny, dt, rho, mi, U_bc)
        result = run_simulation(graph, final_time)
    
        U = np.array(result.ns_x_mesh.phi)
        V = np.array(result.ns_y_mesh.phi)
        if SAVE_RESULTS:
            np.savetxt('U.txt', np.array(U))
            np.savetxt('V.txt', np.array(V))

    if PLOT_RESULTS:
        U = U.reshape(nx, ny - 1)
        V = V.reshape(nx - 1, ny)

        U = np.c_[[0.0] * nx, U, [0.0] * ny]
        U = (U[:,1:] + U[:,:-1]) / 2.0

        V = np.r_[[[0.0] * nx], V, [[0.0] * ny]]
        V = (V[1:,:] + V[:-1,:]) / 2.0

        X, Y = np.meshgrid(np.arange(0.0, size_x, size_x / nx), np.arange(0.0, size_y, size_y / ny))
        plt.figure()
        plt.title("U and V Interpolated on the center of Pressure control volumes")
        plt.quiver(X, Y, U, V)
        plt.show()
