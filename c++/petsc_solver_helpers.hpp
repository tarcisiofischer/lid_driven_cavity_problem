#ifndef C___PETSC_SOLVER_HELPERS_HPP_
#define C___PETSC_SOLVER_HELPERS_HPP_

#include <petsc.h>

class SolverHelper {
public:
	~SolverHelper();
	void setup_options();
	void setup_snes(int nx, int ny);

private:
	PetscOptions _options;
	SNES _snes;
	DM _dm;
	Vec _residual;
};

#endif /* C___PETSC_SOLVER_HELPERS_HPP_ */
