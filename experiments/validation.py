from lid_driven_cavity_problem.nonlinear_solver import petsc_solver_wrapper
from lid_driven_cavity_problem.residual_function import pure_python_residual_function, \
    numpy_residual_function, cython_residual_function, numba_residual_function
from lid_driven_cavity_problem.staggered_grid import Graph
from lid_driven_cavity_problem.time_stepper import run_simulation
import matplotlib.pyplot as plt
import numpy as np


grid_size = 129
Re = 400.0
size_x = 1.0
size_y = 1.0
nx = grid_size
ny = grid_size
dt = 1e-2
rho = 1.0
final_time = None  # Run until steady state
mi = 1.0
U_bc = (mi * Re) / (rho * size_x)


def run_solver_and_return_results(language, interpolation_type):
    solver = petsc_solver_wrapper._PetscSolverWrapper()

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

    graph = Graph(size_x, size_y, nx, ny, dt, rho, mi, U_bc)
    if interpolation_type == 'cds':
        graph.use_cds = True
    g = run_simulation(graph, final_time, solver, residual_f)

    U, V, P = \
        np.array(g.ns_x_mesh.phi).reshape(nx, ny - 1), \
        np.array(g.ns_y_mesh.phi).reshape(nx - 1, ny), \
        np.array(g.pressure_mesh.phi)

    del g
    del solver

    return U, V, P


results = {}
for language in [
#     'python',
#     'numpy',
#     'cython',
#     'c++',
    'numba',
]:
    for interpolation_type in ['cds', 'uds']:
        print("Running %s %s..." % (language, interpolation_type,))
        results['%s interpolation' % (interpolation_type,)] = run_solver_and_return_results(language, interpolation_type)

print("All done. Preparing plot...")

U_ghia = np.loadtxt('ghia_ghia_shin_results/U.txt')
V_ghia = np.loadtxt('ghia_ghia_shin_results/V.txt')
pos_U_ghia = U_ghia[:, 0]
U_ghia = U_ghia[:, 2]
pos_V_ghia = V_ghia[:, 0]
V_ghia = V_ghia[:, 2]

plt.figure(1)
plt.title("U velocity in the mesh center-x")
ax = plt.gca()
ax.set_xlabel('U (m/s)')
ax.set_ylabel('y (m)')
for language, (U, V, P) in results.items():
    U_normalized = U / U_bc
    U_center = U_normalized[:, len(U) // 2]
    plt.plot(U_center, np.linspace(0.0, size_y, len(U_center)), label=language)
plt.plot(U_ghia, pos_U_ghia, 'xb', label='Ghia, Ghia and Shin')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure(2)
plt.title("V velocity in the mesh center-y")
ax = plt.gca()
ax.set_xlabel('x (m)')
ax.set_ylabel('V (m/s)')
for language, (U, V, P) in results.items():
    V_normalized = V / U_bc
    V_center = V_normalized[len(V) // 2, :]
    plt.plot(np.linspace(0.0, size_x, len(V_center)), V_center, label=language)
plt.plot(pos_V_ghia, V_ghia, 'xb', label='Ghia, Ghia and Shin')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

print("All done.")
plt.show()
