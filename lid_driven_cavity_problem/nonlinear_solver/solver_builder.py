PETSC_SOLVER = 'petsc'
SCIPY_SOLVER = 'scipy'

def build(solver_name, residual_f):
    if solver_name == PETSC_SOLVER:
        from lid_driven_cavity_problem.nonlinear_solver import petsc_solver_wrapper
        solver_class = petsc_solver_wrapper.PetscSolverWrapper
    elif solver_name == SCIPY_SOLVER:
        from lid_driven_cavity_problem.nonlinear_solver import scipy_solver_wrapper
        solver_class = scipy_solver_wrapper.ScipySolverWrapper
    else:
        assert False, "Unknown solver named %s" % (solver_name,)

    solver_wrapper = solver_class(residual_f)
    return solver_wrapper.solve
