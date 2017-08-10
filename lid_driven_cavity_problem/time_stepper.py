from lid_driven_cavity_problem import newton_solver
from copy import copy



def run_simulation(graph, final_time):
    t = 0.0
    while t <= final_time:
        print("time: %s/%s" % (t, final_time))
        new_graph = newton_solver.solve(graph)

        # Copy old solution to new solution
        for mesh_name in ['pressure_mesh', 'ns_x_mesh', 'ns_y_mesh']:
            new_mesh = getattr(new_graph, mesh_name)
            old_mesh = getattr(graph, mesh_name)
            new_mesh.phi_old = copy(old_mesh.phi)
        graph = new_graph
        t += graph.dt
    return graph
