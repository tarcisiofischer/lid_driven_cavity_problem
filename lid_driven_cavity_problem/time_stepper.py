import logging

from lid_driven_cavity_problem.nonlinear_solver import petsc_solver_wrapper
from lid_driven_cavity_problem.nonlinear_solver.exceptions import SolverDivergedException


logger = logging.getLogger(__name__)

def run_simulation(graph, final_time, solver=None, minimum_dt=1e-6, adaptative_dt=True):
    if solver is None:
        wrapper = petsc_solver_wrapper.PetscSolverWrapper()
        solver = wrapper.solve

    t = 0.0
    while final_time is None or t < final_time:
        if final_time is not None and t + graph.dt > final_time:
            graph.dt = final_time - t
            logger.info("graph.dt would override final_time. new graph.dt=%s" % (graph.dt,))

        logger.info("time: %s/%s" % (t, final_time))
        try:
            new_graph = solver(graph)
        except SolverDivergedException:
            logger.info('Simulation diverged.')
            if adaptative_dt:
                graph.dt /= 2.0
                logger.info("Will try with dt=%s" % (graph.dt,))

            if graph.dt <= minimum_dt:
                raise RuntimeError("Timestep has reached a too low value. Giving up.")

            continue

        graph = new_graph
        t += graph.dt

        if final_time == None:
            import numpy as np
            norm = np.linalg.norm(graph.ns_x_mesh.phi - graph.ns_x_mesh.phi_old, ord=np.inf) / graph.bc
            if norm < 1e-6:
                logger.info("Simulation reached Steady State at t=%s (norm=%s)" % (t, norm,))
                break
            else:
                logger.info("norm=%s" % (norm,))

            if adaptative_dt:
                graph.dt *= 1.2
        elif abs(final_time - t) <= 1e-5:
            logger.info("Simulation converged.")
            if adaptative_dt:
                graph.dt *= 2.0
                logger.info("Will update dt=%s" % (graph.dt,))
        else:
            logger.info("Simulation converged. Finished at t=%s" % (t,))

    return graph
