from copy import copy, deepcopy
import logging

from lid_driven_cavity_problem.nonlinear_solver import petsc_solver_wrapper
from lid_driven_cavity_problem.nonlinear_solver.exceptions import SolverDivergedException
from lid_driven_cavity_problem.residual_function import numpy_residual_function


logger = logging.getLogger(__name__)

def run_simulation(graph, final_time, solver=None, residual_f=None, minimum_dt=1e-6):
    if solver is None:
        solver = petsc_solver_wrapper.solve
    if residual_f is None:
        residual_f = numpy_residual_function.residual_function

    t = 0.0
    while final_time is None or t < final_time:
        if final_time is not None and t + graph.dt > final_time:
            graph.dt = final_time - t
            logger.info("graph.dt would override final_time. new graph.dt=%s" % (graph.dt,))

        logger.info("time: %s/%s" % (t, final_time))
        try:
            new_graph = solver(graph, residual_f)
        except SolverDivergedException:
            graph.dt /= 2.0

            if graph.dt <= minimum_dt:
                raise RuntimeError("Timestep has reached a too low value. Giving up.")

            logger.info("Simulation diverged. Will try with dt=%s" % (graph.dt,))
            continue

        graph = new_graph
        t += graph.dt

        if final_time == None:
            import numpy as np
            norm = np.linalg.norm(graph.ns_x_mesh.phi - graph.ns_x_mesh.phi_old, ord=np.inf) / graph.bc
            if norm < 1e-4:
                logger.info("Simulation reached Steady State at t=%s (norm=%s)" % (t, norm,))
                break
            else:
                logger.info("norm=%s" % (norm,))
            graph.dt *= 1.2
        elif t < final_time:
            graph.dt *= 2.0
            logger.info("Simulation converged. Will update dt=%s" % (graph.dt,))
        else:
            logger.info("Simulation converged. Finished at t=%s" % (t,))

    return graph
