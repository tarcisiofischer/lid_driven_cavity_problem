from copy import deepcopy
import logging

from scipy.optimize.minpack import fsolve

from lid_driven_cavity_problem.nonlinear_solver._common import _create_X, _recover_X
from lid_driven_cavity_problem.nonlinear_solver.exceptions import SolverDivergedException
from lid_driven_cavity_problem.options import PLOT_JACOBIAN, SHOW_SOLVER_DETAILS, IGNORE_DIVERGED


logger = logging.getLogger(__name__)


def solve(graph, residual_f):
    new_graph = deepcopy(graph)
    new_graph.ns_x_mesh.phi_old = deepcopy(graph.ns_x_mesh.phi)
    new_graph.ns_y_mesh.phi_old = deepcopy(graph.ns_y_mesh.phi)
    new_graph.pressure_mesh.phi_old = deepcopy(graph.pressure_mesh.phi)

    pressure_mesh = new_graph.pressure_mesh
    ns_x_mesh = new_graph.ns_x_mesh
    ns_y_mesh = new_graph.ns_y_mesh

    # Prepare initial guess
    X = _create_X(ns_x_mesh.phi, ns_y_mesh.phi, pressure_mesh.phi, new_graph)

    if PLOT_JACOBIAN:
        from lid_driven_cavity_problem.nonlinear_solver._utils import _plot_jacobian
        _plot_jacobian(new_graph, X)
        assert False, "Finished plotting Jacobian matrix. Program will be terminated (This is expected behavior)"

    X_, infodict, ier, mesg = fsolve(residual_f, X, args=(new_graph,), full_output=True)
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

    U, V, P = _recover_X(X_, new_graph)
    new_graph.ns_x_mesh.phi = U
    new_graph.ns_y_mesh.phi = V
    new_graph.pressure_mesh.phi = P

    return new_graph
