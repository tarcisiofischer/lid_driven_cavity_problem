from copy import copy
import logging

from lid_driven_cavity_problem import newton_solver
from lid_driven_cavity_problem.newton_solver import SolverDivergedException

logger = logging.getLogger(__name__)

def run_simulation(graph, final_time, solver=None, minimum_dt=1e-4):
    if solver is None:
        solver = newton_solver.solve_using_petsc

    t = 0.0
    while t <= final_time:
        if graph.dt <= minimum_dt:
            raise RuntimeError("Timestep has reached a too low value. Giving up.")

        logger.info("time: %s/%s" % (t, final_time))
        try:
            new_graph = solver(graph)
        except SolverDivergedException:
            graph.dt /= 2.0
            logger.info("Simulation diverged. Will try with dt=%s" % (graph.dt,))
            continue

        # Copy old solution to new solution
        for mesh_name in ['pressure_mesh', 'ns_x_mesh', 'ns_y_mesh']:
            new_mesh = getattr(new_graph, mesh_name)
            old_mesh = getattr(graph, mesh_name)
            new_mesh.phi_old = copy(old_mesh.phi)
        graph = new_graph
        t += graph.dt

        graph.dt *= 2.0
        logger.info("Simulation converged. Will update dt=%s" % (graph.dt,))

    return graph
