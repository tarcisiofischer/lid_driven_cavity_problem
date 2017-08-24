from copy import deepcopy
import logging

from scipy.optimize.minpack import fsolve

from lid_driven_cavity_problem.nonlinear_solver._common import _create_X, _recover_X
from lid_driven_cavity_problem.nonlinear_solver.exceptions import SolverDivergedException
from lid_driven_cavity_problem.options import PLOT_JACOBIAN, SHOW_SOLVER_DETAILS, IGNORE_DIVERGED


logger = logging.getLogger(__name__)


def solve(graph, residual_f):
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh

    U = ns_x_mesh.phi
    V = ns_y_mesh.phi
    P = pressure_mesh.phi
    X = _create_X(U, V, P, graph)

    if PLOT_JACOBIAN:
        from lid_driven_cavity_problem.nonlinear_solver._utils import _plot_jacobian
        _plot_jacobian(graph, X)
        assert False, "Finished plotting Jacobian matrix. Program will be terminated (This is expected behavior)"

    X_, infodict, ier, mesg = fsolve(residual_f, X, args=(graph,), full_output=True)
    if SHOW_SOLVER_DETAILS:
        logger.info("Number of function calls=%s" % (infodict['nfev'],))
        if ier == 1:
            logger.info("Converged")
        else:
            logger.info("Diverged")
            logger.info(mesg)

    if not IGNORE_DIVERGED:
        if not ier == 1:
            raise SolverDivergedException()

    U, V, P = _recover_X(X_, graph)

    new_graph = deepcopy(graph)
    for i in range(len(new_graph.ns_x_mesh)):
        new_graph.ns_x_mesh.phi[i] = U[i]
    for i in range(len(new_graph.ns_y_mesh)):
        new_graph.ns_y_mesh.phi[i] = V[i]
    for i in range(len(new_graph.pressure_mesh)):
        new_graph.pressure_mesh.phi[i] = P[i]

    return new_graph
