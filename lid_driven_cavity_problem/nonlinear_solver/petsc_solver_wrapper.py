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

class PetscSolverWrapper:
    def __init__(self, residual_f):
        self._active_graph = None
        self._residual_f = residual_f
        self._first_run = True
        self._setup_options()

    def residual_function_for_petsc(self, snes, x, f):
        '''
        Wrapper over our `residual_f` so that it's in a way expected by PETSc.
        '''
        x = x[:]  # transforms `PETSc.Vec` into `numpy.ndarray`
        f[:] = self._residual_f(x, self._active_graph)
        f.assemble()

    def solve(self, graph):
        assert self._active_graph is None

        self._active_graph = deepcopy(graph)
        self._active_graph.ns_x_mesh.phi_old = deepcopy(graph.ns_x_mesh.phi)
        self._active_graph.ns_y_mesh.phi_old = deepcopy(graph.ns_y_mesh.phi)
        self._active_graph.pressure_mesh.phi_old = deepcopy(graph.pressure_mesh.phi)

        pressure_mesh = self._active_graph.pressure_mesh
        ns_x_mesh = self._active_graph.ns_x_mesh
        ns_y_mesh = self._active_graph.ns_y_mesh

        # Prepare initial guess
        X = _create_X(ns_x_mesh.phi, ns_y_mesh.phi, pressure_mesh.phi, self._active_graph)
        N = len(X)

        if PLOT_JACOBIAN:
            from lid_driven_cavity_problem.nonlinear_solver._utils import _plot_jacobian
            _plot_jacobian(self._active_graph, X)
            assert False, "Finished plotting Jacobian matrix. Program will be terminated (This is expected behavior)"

        if self._first_run:
            self._setup_snes(pressure_mesh, N)

        logger.info("Initial guess = %s" % (X,))
        x = PETSc.Vec().createSeq(N)  # solution vector
        x.setArray(X)
        b = PETSc.Vec().createSeq(N)  # right-hand side
        b.set(0)
        self._snes.solve(b, x)

        if SHOW_SOLVER_DETAILS:
            logger.info("Number of function calls=%s" % (self._snes.getFunctionEvaluations()))
            if self._snes.reason > 0:
                logger.info("Converged with reason: %s" % (PETSC_NONLINEAR_SOLVER_CONVERGENCE_REASONS[self._snes.reason],))
            else:
                logger.info("Diverged with reason: %s" % (PETSC_NONLINEAR_SOLVER_CONVERGENCE_REASONS[self._snes.reason],))
                if self._snes.reason == -3:
                    logger.info("Linear solver divergence reason code: %s" % (self._snes.ksp.reason,))

        if not IGNORE_DIVERGED:
            if self._snes.reason <= 0:
                raise SolverDivergedException()

        U, V, P = _recover_X(x, self._active_graph)
        self._active_graph.ns_x_mesh.phi = U
        self._active_graph.ns_y_mesh.phi = V
        self._active_graph.pressure_mesh.phi = P

        return_graph = self._active_graph
        self._active_graph = None

        return return_graph

    def _solver_monitor(self, snes, its, fnorm):
        logger.info('  %s Residual function norm %s' % (its, fnorm,))

    def _linear_solver_monitor(self, snes, its, fnorm):
        if its % 50 == 0:
            logger.info('[Linear Solver] %s Residual function norm %s' % (its, fnorm,))

    def _setup_snes(self, pressure_mesh, residual_size):
        # Creates the Jacobian matrix structure.
        j_structure = _calculate_jacobian_mask(pressure_mesh.nx, pressure_mesh.ny, 3)
        logger.info("Jacobian NNZ=%s" % (j_structure.nnz,))
        csr = (j_structure.indptr, j_structure.indices, j_structure.data)
        self._petsc_jacobian = PETSc.Mat().createAIJWithArrays(j_structure.shape, csr)
        self._petsc_jacobian.assemble(assembly=self._petsc_jacobian.AssemblyType.FINAL_ASSEMBLY)

        self._comm = PETSc.COMM_WORLD
        self._dm = PETSc.DMShell().create(comm=self._comm)
        self._dm.setMatrix(self._petsc_jacobian)

        # residual vector
        self._r = PETSc.Vec().createSeq(residual_size)

        self._snes = PETSc.SNES().create(comm=self._comm)
        self._snes.setFunction(self.residual_function_for_petsc, self._r)
        self._snes.setDM(self._dm)
        self._snes.setConvergenceHistory()
        self._snes.setFromOptions()
        self._snes.setUseFD(True)
        self._snes.setMonitor(self._solver_monitor)
        self._snes.ksp.setMonitor(self._linear_solver_monitor)
        self._snes.setTolerances(rtol=1e-4, atol=1e-4, stol=1e-4, max_it=50)

    def _setup_options(self):
        options = PETSc.Options()
        options.clear()
        options.setValue('mat_fd_type', 'ds')
        options.setValue('pc_type', 'ilu')
        options.setValue('pc_factor_shift_type', 'NONZERO')
        options.setValue('ksp_type', 'gmres')
        options.setValue('snes_type', 'newtonls')
        options.setValue('snes_linesearch_type', 'basic')
