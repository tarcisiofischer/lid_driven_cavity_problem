import logging
import sys
import time

from lid_driven_cavity_problem.nonlinear_solver import solver_builder
from lid_driven_cavity_problem.residual_function import numpy_residual_function, \
    pure_python_residual_function, numba_residual_function, \
    cython_residual_function
from lid_driven_cavity_problem.staggered_grid import Graph
from lid_driven_cavity_problem.time_stepper import run_simulation
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

PLOT_RESULTS = True

solver_name = solver_builder.PETSC_SOLVER
language = 'c++_omp'

if language == 'python':
    residual_f = pure_python_residual_function.residual_function
elif language == 'numpy':
    residual_f = numpy_residual_function.residual_function
elif language == 'cython':
    residual_f = cython_residual_function.residual_function
elif language == 'c++':
    from lid_driven_cavity_problem.residual_function import cpp_residual_function
    residual_f = cpp_residual_function.residual_function
elif language == 'c++_omp':
    from lid_driven_cavity_problem.residual_function import cpp_omp_residual_function
    residual_f = cpp_omp_residual_function.residual_function
elif language == 'numba':
    residual_f = numba_residual_function.residual_function
else:
    print("WARNING: Unknown residual function %s. Will use default." % (SOLVER_TYPE,))
    solver = None

solver = solver_builder.build(solver_name, residual_f)

size_x = 1.0
size_y = 1.0
nx = 180
ny = 180
dt = 1e-2
rho = 1.0
final_time = None  # Run until steady state
mi = 1.0
Re = 400.0
U_bc = (mi * Re) / (rho * size_x)
print("Run Parameters:")
print("size_x = %s" % (size_x,))
print("size_y = %s" % (size_y,))
print("nx = %s" % (nx,))
print("ny = %s" % (ny,))
print("dt = %s" % (dt,))
print("rho = %s" % (rho,))
print("mi = %s" % (mi,))
print("U_bc = %s" % (U_bc,))
print("Re = %s" % (Re,))
print("")

graph = Graph(size_x, size_y, nx, ny, dt, rho, mi, U_bc)
b = time.time()
result = run_simulation(graph, final_time, solver)
print(time.time() - b)

U = np.array(result.ns_x_mesh.phi)
V = np.array(result.ns_y_mesh.phi)

U = U.reshape(nx, ny - 1)
V = V.reshape(nx - 1, ny)

U = np.c_[[0.0] * nx, U, [0.0] * ny]
U = (U[:, 1:] + U[:, :-1]) / 2.0

V = np.r_[[[0.0] * nx], V, [[0.0] * ny]]
V = (V[1:, :] + V[:-1, :]) / 2.0

X, Y = np.meshgrid(np.arange(0.0, size_x, size_x / nx), np.arange(0.0, size_y, size_y / ny))
plt.figure(1)
plt.title("Velocity streamlines")
plt.streamplot(X, Y, U, V, color=U, linewidth=2)

plt.figure(2)
plt.title("U and V Interpolated on the center of Pressure control volumes")
plt.quiver(X, Y, U, V)

U = np.array(result.ns_x_mesh.phi)
V = np.array(result.ns_y_mesh.phi)
U = U.reshape(nx, ny - 1)
V = V.reshape(nx - 1, ny)

U_ghia = np.loadtxt('ghia_ghia_shin_results/U.txt')
V_ghia = np.loadtxt('ghia_ghia_shin_results/V.txt')
pos_U_ghia = U_ghia[:, 0]
pos_V_ghia = V_ghia[:, 0]
if np.isclose(Re, 100.0):
    U_ghia = U_ghia[:, 1]
    V_ghia = V_ghia[:, 1]
elif np.isclose(Re, 400.0):
    U_ghia = U_ghia[:, 2]
    V_ghia = V_ghia[:, 2]

plt.figure(3)
plt.title("U velocity in the mesh center-x")
U_normalized = U / U_bc
U_center = U_normalized[:, len(U) // 2]
plt.plot(U_center, np.linspace(0.0, size_y, len(U_center)))
if U_ghia is not None:
    plt.plot(U_ghia, pos_U_ghia, 'xb')

plt.figure(4)
plt.title("V velocity in the mesh center-y")
V_normalized = V / U_bc
V_center = V_normalized[len(V) // 2, :]
plt.plot(np.linspace(0.0, size_x, len(V_center)), V_center)
if V_ghia is not None:
    plt.plot(pos_V_ghia, V_ghia, 'xb')

plt.show()
