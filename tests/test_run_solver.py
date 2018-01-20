import os

import pytest

from lid_driven_cavity_problem.nonlinear_solver import petsc_solver_wrapper, scipy_solver_wrapper
from lid_driven_cavity_problem.staggered_grid import Graph
from lid_driven_cavity_problem.time_stepper import run_simulation
import numpy as np

GENERATE = False

def get_petsc_solver_wrapper():
    wrapper = petsc_solver_wrapper.PetscSolverWrapper()
    return wrapper.solve

@pytest.mark.parametrize(
    ("solver_function"),
    (
        get_petsc_solver_wrapper(),
        scipy_solver_wrapper.solve,
    )
)
def test_small_case(solver_function):
    size_x = 1.0
    size_y = 1.0
    nx = 15
    ny = 15
    dt = 0.05
    rho = 1.0
    final_time = 0.1
    mi = 1.0
    Re = 10
    U_bc = (mi * Re) / (rho * size_x)

    graph = Graph(size_x, size_y, nx, ny, dt, rho, mi, U_bc)
    result = run_simulation(graph, final_time, solver_function)

    U = np.array(result.ns_x_mesh.phi)
    V = np.array(result.ns_y_mesh.phi)

    expected_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_run_solver')
    expected_U_path = os.path.join(expected_path, 'U.txt')
    expected_V_path = os.path.join(expected_path, 'V.txt')
    if GENERATE:
        np.savetxt(expected_U_path, U)
        np.savetxt(expected_V_path, V)
        assert False, "Generation finished. Failing test (This is expected behavior)"

    expected_U = np.loadtxt(expected_U_path)
    expected_V = np.loadtxt(expected_V_path)

    assert np.allclose(U, expected_U, rtol=1e-4, atol=1e-1)
    assert np.allclose(V, expected_V, rtol=1e-4, atol=1e-1)
