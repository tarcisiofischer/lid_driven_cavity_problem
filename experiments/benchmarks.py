import time

from lid_driven_cavity_problem.nonlinear_solver import petsc_solver_wrapper, scipy_solver_wrapper
from lid_driven_cavity_problem.residual_function import pure_python_residual_function, \
    numpy_residual_function, cython_residual_function, numba_residual_function
from lid_driven_cavity_problem.staggered_grid import Graph
from lid_driven_cavity_problem.time_stepper import run_simulation
import matplotlib.pyplot as plt
import numpy as np

N_RUNS = 5

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
    runs = []
    for i in range(N_RUNS):
        print("Run %s/%s" % (i + 1, N_RUNS,))
        start = time.time()
        g = run_simulation(graph, final_time, solver, residual_f)
        stop = time.time()
        runs.append(stop - start)

    del g
    del solver

    return np.mean(runs)

available_grid_sizes = [8, 16, 32, 64, 128]
results = {}
for language in [
    'python',
    'numpy',
    'cython',
    'c++',
    'numba',
]:
    results[language] = []
    for grid_size in available_grid_sizes:
        if grid_size == 128 and language == 'python':
            print("Skipping 128x128 mesh with raw Python, as it'll take too long")
            continue

        print("Running %s@%sx%s" % (language, grid_size, grid_size,))
        results[language].append(run_solver_and_measure_time(language=language, grid_size=grid_size))

plt.figure(1)
plt.title("Time comparison")
ax = plt.gca()
ax.set_xlabel('Grid Size')
ax.set_ylabel('Time (s)')
for language, times in results.items():
    plt.plot(np.arange(0, len(times), 1), np.array(times), label=language)
labels = [item.get_text() for item in ax.get_xticklabels()]
for i, gs in enumerate(available_grid_sizes):
    labels[i] = '%sx%s' % (gs, gs,)
plt.xticks(range(len(available_grid_sizes)))
ax.set_xticklabels(labels)
ax.set_yscale("log")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
