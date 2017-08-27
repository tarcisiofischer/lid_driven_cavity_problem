from lid_driven_cavity_problem.residual_function import pure_python_residual_function, \
    numba_residual_function, cython_residual_function
from lid_driven_cavity_problem.staggered_grid import Graph
import _residual_function
import numpy as np


def test_residual_function():
    x = np.array([
        123.0, 0.002, -0.02,
        123.0, 0.002, -0.02,
        123.0, 0.002, -0.02,
        123.0, 0.002, -0.02,
        123.0, 0.002, -0.02,
        123.0, 0.002, -0.02,
        123.0, 0.002, -0.02,
        123.0, 0.002, -0.02,
        123.0, 0.002, -0.02,
    ])
    graph = Graph(
        1.0,
        1.0,
        3,
        3,
        0.01,
        1.0,
        1.0,
        100.0,
        initial_P=100.0,
        initial_U=0.001,
        initial_V=-0.01,
    )

    reference_results = pure_python_residual_function.residual_function(x, graph)
    for f in [
        _residual_function.residual_function,
        numba_residual_function.residual_function,
        cython_residual_function.residual_function,
    ]:
        assert np.allclose(f(x, graph), reference_results, rtol=1e-6, atol=1e-4)
