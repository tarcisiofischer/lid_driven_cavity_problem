from lid_driven_cavity_problem import newton_solver


def run_simulation(graph, final_time, dt):
    t = 0.0
    while t <= final_time:
        graph = newton_solver.solve(graph, dt)
