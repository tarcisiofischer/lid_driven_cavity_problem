from lid_driven_cavity_problem import newton_solver
from copy import copy
from lid_driven_cavity_problem.newton_solver import SolverDivergedException


SHOW_TIMESTEPPER_LOGS = True
MINIMUM_DT = 1e-4

def run_simulation(graph, final_time):
    t = 0.0
    while t <= final_time:
        if graph.dt <= MINIMUM_DT:
            raise RuntimeError("Timestep has reached a too low value. Giving up.")

        if SHOW_TIMESTEPPER_LOGS:
            print("time: %s/%s" % (t, final_time))
        try:
            new_graph = newton_solver.solve(graph)
        except SolverDivergedException:
            graph.dt /= 2.0
            if SHOW_TIMESTEPPER_LOGS:
                print("Simulation diverged. Will try with dt=%s" % (graph.dt,))
            continue

        # Copy old solution to new solution
        for mesh_name in ['pressure_mesh', 'ns_x_mesh', 'ns_y_mesh']:
            new_mesh = getattr(new_graph, mesh_name)
            old_mesh = getattr(graph, mesh_name)
            new_mesh.phi_old = copy(old_mesh.phi)
        graph = new_graph
        t += graph.dt
        graph.dt *= 2.0

    return graph
