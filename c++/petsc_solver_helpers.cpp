#include "petsc_solver_helpers.hpp"
#include <iostream>

SolverHelper::~SolverHelper() {
	PetscOptionsDestroy(&this->_options);
}

void SolverHelper::setup_snes(int nx, int ny)
{
    auto N = nx * ny * 3;

    // TODO
//    // Creates the Jacobian matrix structure.
//    auto j_structure = _calculate_jacobian_mask(nx, ny, 3);
//    csr = (j_structure.indptr, j_structure.indices, j_structure.data)
//    self._petsc_jacobian = PETSc.Mat().createAIJWithArrays(j_structure.shape, csr)
//    self._petsc_jacobian.assemble(assembly=self._petsc_jacobian.AssemblyType.FINAL_ASSEMBLY)

//	self._comm = PETSc.COMM_WORLD
    auto const& comm = PETSC_COMM_WORLD;

//    self._dm = PETSc.DMShell().create(comm=self._comm)
    DMShellCreate(comm, &this->_dm);
    // TODO
//    self._dm.setMatrix(self._petsc_jacobian)
//    DMShellSetMatrix(this->_dm, petsc_jacobian);

    // TODO
//    self._r = PETSc.Vec().createSeq(N)
    VecCreate(comm, &this->_residual);

//    self._snes = PETSc.SNES().create(comm=self._comm)
    SNESCreate(comm, &this->_snes);
    // TODO
//    self._snes.setFunction(self.residual_function_for_petsc, self._r)
//    SNESSetFunction(this->residual_function_for_petsc, this->_residual);

//    self._snes.setDM(self._dm)
    SNESSetDM(this->_snes, this->_dm);
    // TODO
//    self._snes.setConvergenceHistory()
	SNESSetFromOptions(this->_snes);
    // TODO
//    self._snes.setUseFD(True)
    // TODO
//    self._snes.setMonitor(self._solver_monitor)
    // TODO
//    self._snes.ksp.setMonitor(self._linear_solver_monitor)
//    self._snes.setTolerances(rtol=1e-4, atol=1e-4, stol=1e-4, max_it=50)
	SNESSetTolerances(this->_snes, 1e-4, 1e-4, PETSC_DEFAULT, 50, PETSC_DEFAULT);
}

void SolverHelper::setup_options()
{
	PetscOptionsCreate(&this->_options);
	PetscOptionsClear(this->_options);
	PetscOptionsSetValue(this->_options, "mat_fd_type", "ds");
	PetscOptionsSetValue(this->_options, "pc_type", "ilu");
	PetscOptionsSetValue(this->_options, "pc_factor_shift_type", "NONZERO");
	PetscOptionsSetValue(this->_options, "ksp_type", "gmres");
	PetscOptionsSetValue(this->_options, "snes_type", "newtonls");
	PetscOptionsSetValue(this->_options, "snes_linesearch_type", "basic");
}
