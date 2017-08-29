#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array residual_function(py::array X, py::object graph);
py::array residual_function_omp(py::array X, py::object graph);

PYBIND11_PLUGIN(_residual_function) {
    py::module m("_residual_function");
    m.def("residual_function", &residual_function);
    m.def("residual_function_omp", &residual_function_omp);
    return m.ptr();
}
