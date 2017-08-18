from lid_driven_cavity_problem.residual_function import residual_function
from copy import deepcopy
from scipy.optimize.minpack import fsolve
from scipy.optimize.slsqp import approx_jacobian
import numpy as np
from lid_driven_cavity_problem.options import SOLVE_WITH_CLOSE_UVP,\
    FULL_JACOBIAN, PLOT_JACOBIAN, SHOW_SOLVER_DETAILS, IGNORE_DIVERGED


class SolverDivergedException(RuntimeError):
    pass


def _create_X(U, V, P, graph):
    if SOLVE_WITH_CLOSE_UVP:
        X = U + V + P
    else:
        X = np.zeros(shape=(3 * len(P),))
        X[0::3] = P

        extended_U = np.ones(shape=(len(P),))
        nx = graph.ns_x_mesh.nx
        ny = graph.ns_x_mesh.ny
        U_idxs = np.arange(0, len(graph.ns_x_mesh)) + np.repeat(np.arange(0, ny), nx)
        extended_U[U_idxs] = U
        X[1::3] = extended_U

        extended_V = np.r_[V, np.ones(shape=(len(P) - len(V),))]
        X[2::3] = extended_V

    return X


def _recover_X(X, graph):
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh

    if SOLVE_WITH_CLOSE_UVP:
        U = X[0:len(ns_x_mesh)]
        V = X[len(ns_x_mesh):len(ns_x_mesh) + len(ns_y_mesh)]
        P = X[len(ns_x_mesh) + len(ns_y_mesh):len(ns_x_mesh) + len(ns_y_mesh) + len(pressure_mesh)]
    else:
        P = X[0::3][0:len(pressure_mesh)]

        extended_U = X[1::3]
        nx = graph.ns_x_mesh.nx
        ny = graph.ns_x_mesh.ny
        U_idxs = np.arange(0, len(graph.ns_x_mesh)) + np.repeat(np.arange(0, ny), nx)
        U = extended_U[U_idxs]

        V = X[2::3][0:len(ns_y_mesh)]
    return U, V, P


__j_structure_cache = None
def _calculate_jacobian_mask(N, graph):
    global __j_structure_cache
    if __j_structure_cache is not None:
        return __j_structure_cache

    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh
    
    # Must have values in diagonal because of ILU preconditioner
    j_structure = np.zeros((N, N), dtype=bool)

    for i in range(N):
        j_structure[i,i] = 1.0

    # Creates the Jacobian 4 times, because of upwind asymmetries
    for u_sign in [-1.0, 1.0]:
        for v_sign in [-1.0, 1.0]:
            fake_U = [u_sign * 1e-2 * (i + 1) ** 2 for i in range(len(ns_x_mesh.phi))]
            fake_V = [v_sign * 1e-2 * (i + 1) ** 2 for i in range(len(ns_y_mesh.phi))]
            fake_P = [1e3 * (i + 1) ** 2 for i in range(len(pressure_mesh.phi))]
            fake_X = _create_X(fake_U, fake_V, fake_P, graph)
            current_j_structure = approx_jacobian(fake_X, residual_function, 1e-4, graph).astype(dtype='bool')
            j_structure = np.bitwise_or(j_structure, current_j_structure)
    
    __j_structure_cache = j_structure
    return j_structure


def _plot_jacobian(graph, X):
    PLOT_LINES_ON_VARIABLES = False
    PLOT_LINES_ON_ELEMENTS = True
#     PLOT_ONLY = 'U'
#     PLOT_ONLY = 'V'
#     PLOT_ONLY = 'P'
    PLOT_ONLY = None
    
    import matplotlib.pyplot as plt
    J = approx_jacobian(X, residual_function, 1e-4, graph)
    J = J.astype(dtype='bool')
    if PLOT_ONLY == 'U':
        J = J[0::3,:]
    if PLOT_ONLY == 'V':
        J = J[1::3,:]
    if PLOT_ONLY == 'P':
        J = J[2::3,:]
    if PLOT_ONLY is not None:
        J = J[:,0::3] + J[:,1::3] + J[:,2::3]
    plt.imshow(J, aspect='equal', interpolation='none')
    ax = plt.gca()

    if PLOT_LINES_ON_VARIABLES:
        ax.set_xticks(np.arange(-.5, len(J), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(J), 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    if PLOT_LINES_ON_ELEMENTS:
        ax.set_xticks(np.arange(-.5, len(J), 3), minor=False)
        ax.set_yticks(np.arange(-.5, len(J), 3), minor=False)
        ax.grid(color='r', linestyle='-', linewidth=2)

    plt.show()


def solve(graph):
    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh

    U = ns_x_mesh.phi
    V = ns_x_mesh.phi
    P = pressure_mesh.phi
    X = _create_X(U, V, P, graph)

    if PLOT_JACOBIAN:
        _plot_jacobian(graph, X)

    X_, infodict, ier, mesg = fsolve(residual_function, X, args=(graph,), full_output=True)
    if SHOW_SOLVER_DETAILS:
        print("Number of function calls=%s" % (infodict['nfev'],))
        if ier == 1:
            print("Converged")
        else:
            print("Diverged")
            print(mesg)

    if not IGNORE_DIVERGED:
        if not ier == 1:
            raise SolverDivergedException()

    U, V, P = _recover_X(X_, graph)
 
    new_graph = deepcopy(graph)
    for i in range(len(new_graph.ns_x_mesh)):
        new_graph.ns_x_mesh.phi[i] = U[i]
    for i in range(len(new_graph.ns_y_mesh)):
        new_graph.ns_y_mesh.phi[i] = V[i]
    for i in range(len(new_graph.pressure_mesh)):
        new_graph.pressure_mesh.phi[i] = P[i]
 
    return new_graph



from petsc4py import PETSc
def solve_using_petsc(graph):
    def residual_function_for_petsc(snes, x, f):
        '''
        Wrapper over our `residual_function` so that it's in a way expected by PETSc.
        '''
        x = x[:]  # transforms `PETSc.Vec` into `numpy.ndarray`
        f[:] = residual_function(x, graph)
        f.assemble()

    options = PETSc.Options()
    options.clear()

#     options.setValue('snes_test_err', '1e-4')
#     options.setValue('mat_fd_coloring_err', '1e-4')
#     options.setValue('mat_fd_coloring_umin', '1e-4')
#     options.setValue('mat_fd_coloring_view', '::ascii_info')
#     options.setValue('ksp_max_it', '300')
#     options.setValue('ksp_atol', '1e-5')
#     options.setValue('ksp_rtol ', '1e-5')
#     options.setValue('ksp_divtol ', '1e-5')

#     options.setValue('log_view', '')
#     options.setValue('log_all', '')

#     options.setValue('mat_fd_type', 'ds')
    options.setValue('mat_fd_type', 'wp')

    options.setValue('pc_type', 'none')
#     options.setValue('pc_type', 'ilu')
#     options.setValue('pc_type', 'lu')
#     options.setValue('pc_factor_shift_type', 'NONZERO')
#     options.setValue('pc_type', 'svd')

#     options.setValue('ksp_type', 'preonly')
#     options.setValue('ksp_type', 'gmres')
    options.setValue('ksp_type', 'lsqr')

    options.setValue('snes_type', 'newtonls')
#     options.setValue('snes_type', 'qn')
#     options.setValue('snes_type', 'test')
#     options.setValue('snes_type', 'nrichardson')
#     options.setValue('snes_type', 'ksponly')
#     options.setValue('snes_type', 'ngmres')
#     options.setValue('snes_type', 'anderson')
#     options.setValue('snes_type', 'composite')

#     options.setValue('snes_qn_type', 'broyden')
#     options.setValue('snes_qn_type', 'lbfgs')

#     options.setValue('snes_linesearch_type', 'bt')
#     options.setValue('snes_linesearch_type', 'nleqerr')
    options.setValue('snes_linesearch_type', 'basic')
#     options.setValue('snes_linesearch_type', 'l2')
#     options.setValue('snes_linesearch_type', 'cp')

    pressure_mesh = graph.pressure_mesh
    ns_x_mesh = graph.ns_x_mesh
    ns_y_mesh = graph.ns_y_mesh
    X = _create_X(ns_x_mesh.phi, ns_y_mesh.phi, pressure_mesh.phi, graph)

    if PLOT_JACOBIAN:
        _plot_jacobian(graph, X)

    # Creates the Jacobian matrix structure.
    COMM = PETSc.COMM_WORLD
    N = len(X)
    J = PETSc.Mat().createAIJ(N, comm=COMM)
    J.setPreallocationNNZ(N)

    print("Building J...")
    if FULL_JACOBIAN:
        for i in range(N):
            for j in range(N):
                J.setValue(i, j, 0.0)
    else:
        j_structure = _calculate_jacobian_mask(N, graph)

        for i, j in zip(*np.nonzero(j_structure)):
            J.setValue(i, j, 1.0)
    print("Done.")


    J.setUp()
    J.assemble()

    dm = PETSc.DMShell().create(comm=COMM)
    dm.setMatrix(J)

    snes = PETSc.SNES().create(comm=COMM)
    r = PETSc.Vec().createSeq(N)  # residual vector
    x = PETSc.Vec().createSeq(N)  # solution vector
    b = PETSc.Vec().createSeq(N)  # right-hand side
    snes.setFunction(residual_function_for_petsc, r)
    snes.setDM(dm)

    snes.setUseFD(True)

    print("Initial guess = %s" % (X,))
    x.setArray(X)
    b.set(0)

    snes.setConvergenceHistory()
    snes.setFromOptions()

    def _solver_monitor(snes, its, fnorm):
        print('  %s Residual function norm %s' % (its, fnorm,))
    snes.setMonitor(_solver_monitor)

    snes.setTolerances(rtol=1e-4, atol=1e-4, stol=1e-4, max_it=50)
    snes.solve(b, x)
#     rh, ih = snes.getConvergenceHistory()

#     print('(residual, number of linear iterations)')
#     print('\n'.join(str(h) for h in zip(rh, ih)))

    if SHOW_SOLVER_DETAILS:
        print("Number of function calls=%s" % (snes.getFunctionEvaluations()))
        
        REASONS = {
            0: 'still iterating',
            # Converged
            2: '||F|| < atol',
            3: '||F|| < rtol',
            4: 'Newton computed step size small; || delta x || < stol || x ||',
            5: 'maximum iterations reached',
            7: 'trust region delta',
            # Diverged
            -1: 'the new x location passed to the function is not in the domain of F',
            -2: 'maximum function count reached',
            -3: 'the linear solve failed',
            -4: 'norm of F is NaN',
            - 5: 'maximum iterations reached',
            -6: 'the line search failed',
            -7: 'inner solve failed',
            -8: '|| J^T b || is small, implies converged to local minimum of F()',
        }
        
        if snes.reason > 0:
            print("Converged with reason:", REASONS[snes.reason])
        else:
            print("Diverged with reason:", REASONS[snes.reason])
            if snes.reason == -3:
                print("Linear solver divergence reason code:", snes.ksp.reason)

    if not IGNORE_DIVERGED:
        if snes.reason <= 0:
            raise SolverDivergedException()

    U, V, P = _recover_X(x, graph)

    new_graph = deepcopy(graph)
    for i in range(len(new_graph.ns_x_mesh)):
        new_graph.ns_x_mesh.phi[i] = U[i]
    for i in range(len(new_graph.ns_y_mesh)):
        new_graph.ns_y_mesh.phi[i] = V[i]
    for i in range(len(new_graph.pressure_mesh)):
        new_graph.pressure_mesh.phi[i] = P[i]
 
    return new_graph
