from copy import copy
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
    while t < final_time:
        if t + graph.dt > final_time:
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

        # Copy old solution to new solution
        for mesh_name in ['pressure_mesh', 'ns_x_mesh', 'ns_y_mesh']:
            new_mesh = getattr(new_graph, mesh_name)
            old_mesh = getattr(graph, mesh_name)
            new_mesh.phi_old = copy(old_mesh.phi)
        graph = new_graph
        t += graph.dt

        if t < final_time:
            graph.dt *= 2.0
            logger.info("Simulation converged. Will update dt=%s" % (graph.dt,))
        else:
            logger.info("Simulation converged. Finished at t=%s" % (t,))

    return graph
