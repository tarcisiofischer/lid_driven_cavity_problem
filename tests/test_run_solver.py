import os

import pytest

from lid_driven_cavity_problem.newton_solver import solve_using_petsc, solve_using_scipy
from lid_driven_cavity_problem.staggered_grid import Graph
from lid_driven_cavity_problem.time_stepper import run_simulation
import numpy as np


@pytest.mark.parametrize(
    ("solver_function"),
    (
        solve_using_petsc,
        solve_using_scipy,
    )
)
def test_small_case(solver_function):
    size_x = 1.0
    size_y = 1.0
    nx = 11
    ny = 11
    dt = 1e-2
    rho = 1.0
    final_time = 0.1
    mi = 1.0
    Re = 100
    U_bc = (mi * Re) / (rho * size_x)

    graph = Graph(size_x, size_y, nx, ny, dt, rho, mi, U_bc)
    result = run_simulation(graph, final_time, solver_function)

    U = np.array(result.ns_x_mesh.phi)
    V = np.array(result.ns_y_mesh.phi)

    expected_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_run_solver')
    expected_U = np.loadtxt(os.path.join(expected_path, 'U.txt'))
    expected_V = np.loadtxt(os.path.join(expected_path, 'V.txt'))

    assert np.allclose(U, expected_U, rtol=1e-4, atol=1e-6)
    assert np.allclose(V, expected_V, rtol=1e-4, atol=1e-6)
