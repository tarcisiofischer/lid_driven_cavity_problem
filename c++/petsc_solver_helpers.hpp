#ifndef C___PETSC_SOLVER_HELPERS_HPP_
#define C___PETSC_SOLVER_HELPERS_HPP_

#include <petsc.h>

class SolverHelper {
public:
	SolverHelper();
	~SolverHelper();
	void setup_options();

private:
	PetscOptions *options;
};

#endif /* C___PETSC_SOLVER_HELPERS_HPP_ */
