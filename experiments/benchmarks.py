import time

from lid_driven_cavity_problem.nonlinear_solver import petsc_solver_wrapper, scipy_solver_wrapper
from lid_driven_cavity_problem.residual_function import pure_python_residual_function, \
    numpy_residual_function, cython_residual_function, numba_residual_function
from lid_driven_cavity_problem.staggered_grid import Graph
from lid_driven_cavity_problem.time_stepper import run_simulation


def run_solver_and_measure_time(solver_type='petsc', language='c++', grid_size=32, Re=400.0):
    if solver_type == 'petsc':
        solver = petsc_solver_wrapper._PetscSolverWrapper()
    elif solver_type == 'scipy':
        solver = scipy_solver_wrapper.solve
    else:
        assert False

    if language == 'python':
        residual_f = pure_python_residual_function.residual_function
    elif language == 'numpy':
        residual_f = numpy_residual_function.residual_function
    elif language == 'cython':
        residual_f = cython_residual_function.residual_function
    elif language == 'c++':
        from lid_driven_cavity_problem.residual_function import cpp_residual_function
        residual_f = cpp_residual_function.residual_function
    elif language == 'numba':
        residual_f = numba_residual_function.residual_function
    else:
        assert False

    size_x = 1.0
    size_y = 1.0
    nx = grid_size
    ny = grid_size
    dt = 1e-2
    rho = 1.0
    final_time = None  # Run until steady state
    mi = 1.0
    U_bc = (mi * Re) / (rho * size_x)

    graph = Graph(size_x, size_y, nx, ny, dt, rho, mi, U_bc)
    b = time.time()
    g = run_simulation(graph, final_time, solver, residual_f)

    del g
    del solver

    print(time.time() - b)

for language in [
#     'python',
    'numpy',
    'cython',
    'c++',
    'numba',
]:
    for grid_size in [8, 16, 32, 64, 128]:
        print("%s@%s" % (language, grid_size,))
        run_solver_and_measure_time(language=language, grid_size=grid_size)
        print("")
