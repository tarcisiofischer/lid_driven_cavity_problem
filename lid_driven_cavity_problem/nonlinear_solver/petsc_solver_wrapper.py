from copy import deepcopy
import logging

from petsc4py import PETSc

from lid_driven_cavity_problem.nonlinear_solver._common import _create_X, _recover_X, \
    _calculate_jacobian_mask
from lid_driven_cavity_problem.nonlinear_solver.exceptions import SolverDivergedException
from lid_driven_cavity_problem.options import PLOT_JACOBIAN, SHOW_SOLVER_DETAILS, IGNORE_DIVERGED


PETSC_NONLINEAR_SOLVER_CONVERGENCE_REASONS = {
    0: 'still iterating',

    # Converged
    2: '||F|| < atol',
    3: '||F|| < rtol',
    4: 'Newton computed step size small; || delta x || < stol || x ||',
    5: 'maximum iterations reached',
    7: 'trust region delta',

    # Diverged
    - 1: 'the new x location passed to the function is not in the domain of F',
    - 2: 'maximum function count reached',
    - 3: 'the linear solver failed',
    - 4: 'norm of F is NaN',
    - 5: 'maximum iterations reached',
    - 6: 'the line search failed',
    - 7: 'inner solve failed',
    - 8: '|| self._petsc_jacobian^T b || is small, implies converged to local minimum of F()',
}

logger = logging.getLogger(__name__)

class _PetscSolverWrapper(object):
    def __init__(self):
        # Cache for the PETSC Jacobian object
        self._petsc_jacobian = None

        options = PETSc.Options()
        options.clear()

#         options.setValue('snes_test_err', '1e-4')
#         options.setValue('mat_fd_coloring_err', '1e-4')
#         options.setValue('mat_fd_coloring_umin', '1e-4')
#         options.setValue('mat_fd_coloring_view', '::ascii_info')
#         options.setValue('ksp_max_it', '300')
#         options.setValue('ksp_atol', '1e-5')
#         options.setValue('ksp_rtol ', '1e-5')
#         options.setValue('ksp_divtol ', '1e-5')

#         options.setValue('log_view', '')
#         options.setValue('log_all', '')

        options.setValue('mat_fd_type', 'ds')
#         options.setValue('mat_fd_type', 'wp')

#         options.setValue('pc_type', 'none')
        options.setValue('pc_type', 'ilu')
#         options.setValue('pc_type', 'lu')
        options.setValue('pc_factor_shift_type', 'NONZERO')
#         options.setValue('pc_type', 'svd')

#         options.setValue('ksp_type', 'preonly')
        options.setValue('ksp_type', 'gmres')
#         options.setValue('ksp_type', 'lsqr')

        options.setValue('snes_type', 'newtonls')
#         options.setValue('snes_type', 'qn')
#         options.setValue('snes_type', 'test')
#         options.setValue('snes_type', 'nrichardson')
#         options.setValue('snes_type', 'ksponly')
#         options.setValue('snes_type', 'ngmres')
#         options.setValue('snes_type', 'anderson')
#         options.setValue('snes_type', 'composite')

#         options.setValue('snes_qn_type', 'broyden')
#         options.setValue('snes_qn_type', 'lbfgs')

#         options.setValue('snes_linesearch_type', 'bt')
#         options.setValue('snes_linesearch_type', 'nleqerr')
        options.setValue('snes_linesearch_type', 'basic')
#         options.setValue('snes_linesearch_type', 'l2')
#         options.setValue('snes_linesearch_type', 'cp')

    def __call__(self, graph, residual_f):
        new_graph = deepcopy(graph)
        new_graph.ns_x_mesh.phi_old = deepcopy(graph.ns_x_mesh.phi)
        new_graph.ns_y_mesh.phi_old = deepcopy(graph.ns_y_mesh.phi)
        new_graph.pressure_mesh.phi_old = deepcopy(graph.pressure_mesh.phi)

        def residual_function_for_petsc(snes, x, f):
            '''
            Wrapper over our `residual_f` so that it's in a way expected by PETSc.
            '''
            x = x[:]  # transforms `PETSc.Vec` into `numpy.ndarray`
            f[:] = residual_f(x, new_graph)
            f.assemble()

        pressure_mesh = new_graph.pressure_mesh
        ns_x_mesh = new_graph.ns_x_mesh
        ns_y_mesh = new_graph.ns_y_mesh

        # Prepare initial guess
        X = _create_X(ns_x_mesh.phi, ns_y_mesh.phi, pressure_mesh.phi, new_graph)

        if PLOT_JACOBIAN:
            from lid_driven_cavity_problem.nonlinear_solver._utils import _plot_jacobian
            _plot_jacobian(new_graph, X)
            assert False, "Finished plotting Jacobian matrix. Program will be terminated (This is expected behavior)"

        COMM = PETSc.COMM_WORLD
        N = len(X)

        if self._petsc_jacobian is None:
            # Creates the Jacobian matrix structure.
            j_structure = _calculate_jacobian_mask(pressure_mesh.nx, pressure_mesh.ny, 3)
            logger.info("Jacobian NNZ=%s" % (j_structure.nnz,))
            csr = (j_structure.indptr, j_structure.indices, j_structure.data)
            self._petsc_jacobian = PETSc.Mat().createAIJWithArrays(j_structure.shape, csr)
            self._petsc_jacobian.assemble(assembly=self._petsc_jacobian.AssemblyType.FINAL_ASSEMBLY)

        dm = PETSc.DMShell().create(comm=COMM)
        dm.setMatrix(self._petsc_jacobian)

        snes = PETSc.SNES().create(comm=COMM)
        r = PETSc.Vec().createSeq(N)  # residual vector
        x = PETSc.Vec().createSeq(N)  # solution vector
        b = PETSc.Vec().createSeq(N)  # right-hand side
        snes.setFunction(residual_function_for_petsc, r)
        snes.setDM(dm)

        snes.setUseFD(True)

        logger.info("Initial guess = %s" % (X,))
        x.setArray(X)
        b.set(0)

        snes.setConvergenceHistory()
        snes.setFromOptions()

        def _solver_monitor(snes, its, fnorm):
            logger.info('  %s Residual function norm %s' % (its, fnorm,))
        snes.setMonitor(_solver_monitor)

        snes.setTolerances(rtol=1e-4, atol=1e-4, stol=1e-4, max_it=50)
        snes.solve(b, x)

        if SHOW_SOLVER_DETAILS:
            logger.info("Number of function calls=%s" % (snes.getFunctionEvaluations()))
            if snes.reason > 0:
                logger.info("Converged with reason: %s" % (PETSC_NONLINEAR_SOLVER_CONVERGENCE_REASONS[snes.reason],))
            else:
                logger.info("Diverged with reason: %s" % (PETSC_NONLINEAR_SOLVER_CONVERGENCE_REASONS[snes.reason],))
                if snes.reason == -3:
                    logger.info("Linear solver divergence reason code: %s" % (snes.ksp.reason,))

        if not IGNORE_DIVERGED:
            if snes.reason <= 0:
                raise SolverDivergedException()

        U, V, P = _recover_X(x, new_graph)
        new_graph.ns_x_mesh.phi = U
        new_graph.ns_y_mesh.phi = V
        new_graph.pressure_mesh.phi = P

        return new_graph

solve = _PetscSolverWrapper()
