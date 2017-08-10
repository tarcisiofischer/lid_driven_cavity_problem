from lid_driven_cavity_problem.residual_function import residual_function
from copy import deepcopy
from scipy.optimize.minpack import fsolve



def solve(graph):
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh

    U = ns_x_mesh.phi
    V = ns_x_mesh.phi
    P = pressure_mesh.phi
#     print(U)
#     print(V)
#     print(P)

    X = U + V + P
    X_ = fsolve(residual_function, X, args=(graph,))
#     print("Residual: %s" % (X_,))

    U = X_[0:len(ns_x_mesh)]
    V = X_[len(ns_x_mesh):len(ns_x_mesh) + len(ns_y_mesh)]
    P = X_[len(ns_x_mesh) + len(ns_y_mesh):len(ns_x_mesh) + len(ns_y_mesh) + len(pressure_mesh)]
#     print(U)
#     print(V)
#     print(P)
 
    new_graph = deepcopy(graph)
    for i in range(len(new_graph.ns_x_mesh)):
        new_graph.ns_x_mesh.phi[i] = U[i]
    for i in range(len(new_graph.ns_y_mesh)):
        new_graph.ns_y_mesh.phi[i] = V[i]
    for i in range(len(new_graph.pressure_mesh)):
        new_graph.pressure_mesh.phi[i] = P[i]
 
    return new_graph
