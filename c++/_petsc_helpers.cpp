#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "petsc_solver_helpers.hpp"

namespace py = pybind11;

PYBIND11_PLUGIN(_petsc_helpers) {
    py::module m("_petsc_helpers");

    {
    	typedef SolverHelper T;
    	py::class_<T> (m, "SolverHelper")
			.def(py::init<>())
    		.def("setup_options", &T::setup_options);
    }

    return m.ptr();
}
