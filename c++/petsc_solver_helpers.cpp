#include "petsc_solver_helpers.hpp"
#include <iostream>

SolverHelper::SolverHelper() : options(nullptr) {}
SolverHelper::~SolverHelper() {
	if (this->options != nullptr) {
		PetscOptionsDestroy(this->options);
		this->options = nullptr;
	}
}

void SolverHelper::setup_options()
{
	if (this->options != nullptr) {
		PetscOptionsDestroy(this->options);
		this->options = nullptr;
	}

	PetscOptionsCreate(this->options);
	PetscOptionsClear(*this->options);
	PetscOptionsSetValue(*this->options, "mat_fd_type", "ds");
	PetscOptionsSetValue(*this->options, "pc_type", "ilu");
	PetscOptionsSetValue(*this->options, "pc_factor_shift_type", "NONZERO");
	PetscOptionsSetValue(*this->options, "ksp_type", "gmres");
	PetscOptionsSetValue(*this->options, "snes_type", "newtonls");
	PetscOptionsSetValue(*this->options, "snes_linesearch_type", "basic");
}
