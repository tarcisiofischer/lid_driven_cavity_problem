from lid_driven_cavity_problem.residual_function import residual_function
from copy import deepcopy
from scipy.optimize.minpack import fsolve
from scipy.optimize.slsqp import approx_jacobian
import numpy as np

PLOT_JACOBIAN = False
SHOW_SOLVER_DETAILS = True
IGNORE_DIVERGED = False

class SolverDivergedException(RuntimeError):
    pass

def solve(graph):
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh

    U = ns_x_mesh.phi
    V = ns_x_mesh.phi
    P = pressure_mesh.phi

#     X = np.zeros(shape=(3 * len(P),))
#     X[0::3] = np.r_[U, np.zeros(shape=(len(P) - len(U),))]
#     X[1::3] = np.r_[V, np.zeros(shape=(len(P) - len(V),))]
#     X[2::3] = P

    X = U + V + P

    if PLOT_JACOBIAN:
        import matplotlib.pyplot as plt
        J = approx_jacobian(X, residual_function, 1e-4, graph)
        J = J.astype(dtype='bool')
        plt.imshow(J)
        plt.show()

    X_, infodict, ier, mesg = fsolve(residual_function, X, args=(graph,), full_output=True)
    if SHOW_SOLVER_DETAILS:
        print("Number of function calls=%s" % (infodict['nfev'],))
        if ier == 1:
            print("Converged")
        else:
            print("Diverged")
            print(mesg)

    if not IGNORE_DIVERGED:
        if not ier == 1:
            raise SolverDivergedException()
    
    U = X_[0:len(ns_x_mesh)]
    V = X_[len(ns_x_mesh):len(ns_x_mesh) + len(ns_y_mesh)]
    P = X_[len(ns_x_mesh) + len(ns_y_mesh):len(ns_x_mesh) + len(ns_y_mesh) + len(pressure_mesh)]

#     U = X_[0::3][0:len(ns_x_mesh)]
#     V = X_[1::3][0:len(ns_y_mesh)]
#     P = X_[2::3][0:len(pressure_mesh)]
 
    new_graph = deepcopy(graph)
    for i in range(len(new_graph.ns_x_mesh)):
        new_graph.ns_x_mesh.phi[i] = U[i]
    for i in range(len(new_graph.ns_y_mesh)):
        new_graph.ns_y_mesh.phi[i] = V[i]
    for i in range(len(new_graph.pressure_mesh)):
        new_graph.pressure_mesh.phi[i] = P[i]
 
    return new_graph
